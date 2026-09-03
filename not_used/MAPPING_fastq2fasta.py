#!/usr/bin/env python3
"""
PyCharm-run version: edit USER SETTINGS below and click Run.

This version:
- Uses ONLY R2 FASTQs found in FASTQ_DIR
- Trims each read by finding MOTIF (e.g. 'CAGAGAA') and removing everything before it (and the motif itself)
- Optionally removes the following random hexamer (HEX_LEN)
- Aligns trimmed reads to REF_FASTA with bwa mem (single-end)
- Outputs one CSV or XLSX:
    - __ALL__ summary row: overall percent mapped + trimming stats
    - per-contig mapped read counts and percentages
- Optionally writes all unmapped (trimmed) R2 reads to a separate FASTQ

Requires:
  conda install -c bioconda -c conda-forge bwa samtools pandas openpyxl
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path
import gzip

import pandas as pd

# ==================================
# USER SETTINGS (edit in PyCharm)
# ==================================

# Folder containing your FASTQs (multiple lanes OK). Can be nested if RECURSIVE=True.
FASTQ_DIR = "/home/hanst/PhETRIseq/jrg07PMP/JRG07-Sample-P3"
# FASTQ_DIR = "/home/hanst/PhETRIseq/original"

# Combined reference FASTA (all genomes/contigs in one file)
REF_FASTA = "/home/hanst/PhETRIseq/ref_files/luz19_lkd16_14one_pa01.fa"

# Output file (set .csv OR .xlsx). Single output.
OUT_FILE = "/home/hanst/PhETRIseq/jrg07PMP/JRG07-Sample-P3/mapping_stats.csv"
# OUT_FILE = "/home/hanst/PhETRIseq/original/mapping_stats.csv"

# Output FASTQ for unmapped (trimmed) R2 reads.
# Set to None to disable.
UNMAPPED_R2_FASTQ = "/home/hanst/PhETRIseq/jrg07PMP/JRG07-Sample-P3/R2_unmapped_trimmed.fastq.gz"

# Threads for bwa/samtools
THREADS = 16

# Search for FASTQs recursively under FASTQ_DIR?
RECURSIVE = False

# Include contigs with 0 mapped reads in output?
INCLUDE_ZEROS = False

# Keep intermediate BAMs?
KEEP_BAMS = False

# Optional: where to put temp BAMs (default next to OUT_FILE)
# If None, uses: OUT_FILE + ".tmp_bams" folder
TMP_BAM_DIR = None

# ----------------------------
# TRIMMING SETTINGS
# ----------------------------

# Motif marking end of construct/barcode region, immediately before the random hexamer
MOTIF = "CAGAGAA"

# Remove the random hexamer immediately after the motif?
# If you truly want "only transcript" sequence, set this True.
REMOVE_HEXAMER = True
HEX_LEN = 6

# Minimum length required after trimming; shorter reads are discarded
MIN_LEN_AFTER_TRIM = 20

# If motif not found, discard read (recommended)
DISCARD_IF_NO_MOTIF = False

# Skip files that look already-trimmed (avoid double-processing)
SKIP_IF_NAME_CONTAINS = ("trimmed", ".trim.", ".trimmed.")

# ==================================
# END USER SETTINGS
# ==================================


FASTQ_EXTS = (".fastq.gz", ".fq.gz", ".fastq", ".fq")


def which_or_die(exe: str) -> str:
    path = shutil.which(exe)
    if not path:
        raise SystemExit(
            f"ERROR: '{exe}' not found in PATH.\n"
            f"Install with conda:\n"
            f"  conda install -c bioconda -c conda-forge {exe}\n"
        )
    return path


def run(cmd):
    """Run a command, raising on error."""
    print(f"[cmd] {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True)


def ensure_bwa_index(ref_fa: Path, bwa_exe: str):
    expected = [ref_fa.with_suffix(ref_fa.suffix + ext) for ext in [".amb", ".ann", ".bwt", ".pac", ".sa"]]
    if all(p.exists() for p in expected):
        return
    print(f"[info] BWA index not found for {ref_fa}. Building...", file=sys.stderr)
    run([bwa_exe, "index", str(ref_fa)])


def ensure_faidx(ref_fa: Path, samtools_exe: str):
    fai = Path(str(ref_fa) + ".fai")
    if fai.exists():
        return
    print(f"[info] FASTA .fai not found for {ref_fa}. Building with samtools faidx...", file=sys.stderr)
    run([samtools_exe, "faidx", str(ref_fa)])


def list_fastqs(fastq_dir: Path, recursive: bool):
    if recursive:
        files = [p for p in fastq_dir.rglob("*") if p.is_file() and p.name.endswith(FASTQ_EXTS)]
    else:
        files = [p for p in fastq_dir.iterdir() if p.is_file() and p.name.endswith(FASTQ_EXTS)]
    return sorted(files)


def is_r2_fastq(name: str) -> bool:
    """
    Detect R2 files:
      *_R2_*.fastq.gz
      *_R2.fastq.gz
      *_2.fastq.gz
    """
    if "_R2_" in name:
        return True
    if re.search(r"(?:^|[_\.])R2\.(fastq|fq)(\.gz)?$", name):
        return True
    if re.search(r"(?:^|_)2\.(fastq|fq)(\.gz)?$", name):
        return True
    return False


def open_maybe_gz(path: Path, mode="rt"):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode)


def safe_tag_from_r2(r2: Path):
    tag = r2.name
    tag = re.sub(r"\.(fastq|fq)(\.gz)?$", "", tag)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", tag)


def parse_idxstats(text: str):
    """
    samtools idxstats output columns:
      contig  length  mapped  unmapped
    Last line is: *  0  0  <unmapped_total>
    """
    contig_map = {}
    contig_len = {}
    for line in text.strip().splitlines():
        parts = line.split("\t")
        if len(parts) != 4:
            continue
        contig, length, mapped, unmapped = parts
        if contig == "*":
            continue
        contig_len[contig] = int(length)
        contig_map[contig] = int(mapped)
    return contig_len, contig_map


def trim_read(seq: str, qual: str):
    """
    Trim everything before MOTIF and remove MOTIF itself.
    Optionally remove HEX_LEN bases after motif.
    Returns (new_seq, new_qual, keep_bool, reason_str)
    """
    seq_u = seq.upper()
    idx = seq_u.find(MOTIF)
    if idx == -1:
        if DISCARD_IF_NO_MOTIF:
            return None, None, False, "no_motif"
        return seq, qual, True, "kept_untrimmed_no_motif"

    start = idx + len(MOTIF)
    if REMOVE_HEXAMER:
        start += HEX_LEN

    if start >= len(seq):
        return None, None, False, "too_short"

    new_seq = seq[start:]
    new_qual = qual[start:]

    if len(new_seq) < MIN_LEN_AFTER_TRIM:
        return None, None, False, "too_short"

    return new_seq, new_qual, True, "trimmed"


def align_trimmed_r2_to_bam(r2_path: Path, ref_fa: Path, bam_path: Path, bwa_exe: str, samtools_exe: str):
    """
    Stream-trim R2 FASTQ and pipe directly into:
      bwa mem ref.fa -  | samtools view ... | samtools sort -o bam

    Returns trimming stats dict.
    """
    stats = {
        "raw_reads_r2": 0,
        "kept_reads_after_trim": 0,
        "dropped_no_motif": 0,
        "dropped_too_short": 0,
        "kept_untrimmed_no_motif": 0,
    }

    # Start bwa mem reading from stdin ("-")
    bwa = subprocess.Popen(
        [bwa_exe, "mem", "-t", str(THREADS), str(ref_fa), "-"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,  # use bytes
        bufsize=1024 * 1024,
    )

    # Pipe to samtools view, filtering secondary/supplementary
    view = subprocess.Popen(
        [samtools_exe, "view", "-@", str(THREADS), "-b", "-F", "256", "-F", "2048"],
        stdin=bwa.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
        bufsize=1024 * 1024,
    )
    bwa.stdout.close()

    # Pipe to samtools sort -> bam
    sort = subprocess.Popen(
        [samtools_exe, "sort", "-@", str(THREADS), "-o", str(bam_path), "-"],
        stdin=view.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
        bufsize=1024 * 1024,
    )
    view.stdout.close()

    # Stream FASTQ, trim, write kept reads to bwa.stdin
    try:
        with open_maybe_gz(r2_path, "rt") as fin:
            while True:
                header = fin.readline()
                if not header:
                    break
                seq = fin.readline().rstrip("\n")
                plus = fin.readline()
                qual = fin.readline().rstrip("\n")

                stats["raw_reads_r2"] += 1

                new_seq, new_qual, keep, reason = trim_read(seq, qual)
                if not keep:
                    if reason == "no_motif":
                        stats["dropped_no_motif"] += 1
                    else:
                        stats["dropped_too_short"] += 1
                    continue

                if reason == "kept_untrimmed_no_motif":
                    stats["kept_untrimmed_no_motif"] += 1

                stats["kept_reads_after_trim"] += 1

                # Write FASTQ record to bwa stdin
                bwa.stdin.write(header.encode())
                bwa.stdin.write((new_seq + "\n").encode())
                bwa.stdin.write(plus.encode())
                bwa.stdin.write((new_qual + "\n").encode())

    finally:
        if bwa.stdin:
            bwa.stdin.close()

    # Collect stderr/returncodes
    sort_out, sort_err = sort.communicate()
    view_out, view_err = view.communicate()
    bwa_out, bwa_err = bwa.communicate()

    if bwa.returncode != 0 or view.returncode != 0 or sort.returncode != 0:
        msg = [
            f"ERROR: pipeline failed for {r2_path}",
            f"  bwa returncode={bwa.returncode}",
            f"  view returncode={view.returncode}",
            f"  sort returncode={sort.returncode}",
        ]
        if bwa_err:
            msg.append("---- bwa stderr ----")
            msg.append(bwa_err.decode(errors="replace"))
        if view_err:
            msg.append("---- samtools view stderr ----")
            msg.append(view_err.decode(errors="replace"))
        if sort_err:
            msg.append("---- samtools sort stderr ----")
            msg.append(sort_err.decode(errors="replace"))
        raise SystemExit("\n".join(msg))

    # Index BAM
    run([samtools_exe, "index", str(bam_path)])

    return stats


def main():
    fastq_dir = Path(FASTQ_DIR).expanduser().resolve()
    ref_fa = Path(REF_FASTA).expanduser().resolve()
    out_path = Path(OUT_FILE).expanduser().resolve()

    if not fastq_dir.exists():
        raise SystemExit(f"ERROR: FASTQ_DIR not found: {fastq_dir}")
    if not ref_fa.exists():
        raise SystemExit(f"ERROR: REF_FASTA not found: {ref_fa}")

    bwa = which_or_die("bwa")
    samtools = which_or_die("samtools")

    ensure_bwa_index(ref_fa, bwa)
    ensure_faidx(ref_fa, samtools)

    fastqs = list_fastqs(fastq_dir, RECURSIVE)
    if not fastqs:
        raise SystemExit(f"ERROR: No FASTQ files found in: {fastq_dir}")

    r2_files = []
    for p in fastqs:
        nlow = p.name.lower()
        if any(s in nlow for s in SKIP_IF_NAME_CONTAINS):
            continue
        if is_r2_fastq(p.name):
            r2_files.append(p)

    if not r2_files:
        raise SystemExit("ERROR: No R2 FASTQ files found (looking for *_R2_* or *_2.fastq* patterns).")

    if TMP_BAM_DIR:
        tmp_dir = Path(TMP_BAM_DIR).expanduser().resolve()
    else:
        tmp_dir = out_path.parent / (out_path.stem + ".tmp_bams")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Prepare unmapped reads output (combined across all R2 files)
    if UNMAPPED_R2_FASTQ:
        unmapped_path = Path(UNMAPPED_R2_FASTQ).expanduser().resolve()
        unmapped_path.parent.mkdir(parents=True, exist_ok=True)
        # Start fresh
        if unmapped_path.exists():
            unmapped_path.unlink()
    else:
        unmapped_path = None

    # Aggregates
    total_reads_all = 0              # total reads entering alignment (post-trim kept)
    mapped_reads_all = 0             # mapped reads in BAM
    contig_lengths = {}
    contig_mapped_all = {}

    raw_reads_all = 0
    dropped_no_motif_all = 0
    dropped_too_short_all = 0
    kept_untrimmed_no_motif_all = 0

    print(f"[info] FASTQ_DIR: {fastq_dir}", file=sys.stderr)
    print(f"[info] Found {len(r2_files)} R2 FASTQ file(s) to process.", file=sys.stderr)

    for i, r2 in enumerate(r2_files, start=1):
        tag = safe_tag_from_r2(r2)
        bam = tmp_dir / f"{tag}.primary.bam"

        print(f"[info] ({i}/{len(r2_files)}) Trimming+Mapping R2: {r2.name}", file=sys.stderr)

        trim_stats = align_trimmed_r2_to_bam(r2, ref_fa, bam, bwa, samtools)

        raw_reads_all += trim_stats["raw_reads_r2"]
        total_reads_all += trim_stats["kept_reads_after_trim"]
        dropped_no_motif_all += trim_stats["dropped_no_motif"]
        dropped_too_short_all += trim_stats["dropped_too_short"]
        kept_untrimmed_no_motif_all += trim_stats["kept_untrimmed_no_motif"]

        # Count mapped/unmapped from BAM
        # Total records in BAM equals kept_reads_after_trim (one record per read, mapped or unmapped)
        total_reads_bam = int(subprocess.check_output([samtools, "view", "-c", str(bam)]).decode().strip())
        mapped_reads = int(subprocess.check_output([samtools, "view", "-c", "-F", "4", str(bam)]).decode().strip())
        mapped_reads_all += mapped_reads

        # Sanity check (optional but useful)
        if total_reads_bam != trim_stats["kept_reads_after_trim"]:
            print(
                f"[warn] BAM record count ({total_reads_bam}) != kept_reads_after_trim "
                f"({trim_stats['kept_reads_after_trim']}) for {r2.name}",
                file=sys.stderr
            )

        idx = subprocess.check_output([samtools, "idxstats", str(bam)]).decode()
        lens, mapped_by_contig = parse_idxstats(idx)

        if not contig_lengths:
            contig_lengths = lens

        for contig, mcount in mapped_by_contig.items():
            contig_mapped_all[contig] = contig_mapped_all.get(contig, 0) + mcount

        # Extract unmapped (trimmed) reads and append to global unmapped FASTQ
        if unmapped_path is not None:
            if str(unmapped_path).endswith(".gz"):
                out_handle = gzip.open(unmapped_path, "ab")
            else:
                out_handle = open(unmapped_path, "ab")

            with out_handle as fout:
                p = subprocess.Popen(
                    [samtools, "fastq", "-f", "4", str(bam)],
                    stdout=fout,
                    stderr=subprocess.PIPE,
                )
                _, err = p.communicate()
                if p.returncode != 0:
                    raise SystemExit(
                        f"ERROR: samtools fastq failed on {bam}:\n"
                        f"{err.decode(errors='replace')}"
                    )

    pct_mapped_trimmed = (mapped_reads_all / total_reads_all * 100.0) if total_reads_all else 0.0
    pct_mapped_raw = (mapped_reads_all / raw_reads_all * 100.0) if raw_reads_all else 0.0
    pct_kept = (total_reads_all / raw_reads_all * 100.0) if raw_reads_all else 0.0

    # Output rows
    rows = [{
        "contig": "__ALL__",
        "length_bp": "",
        "mapped_reads_primary": mapped_reads_all,
        "total_reads_primary": total_reads_all,   # NOTE: post-trim kept reads
        "raw_reads_r2": raw_reads_all,
        "dropped_no_motif": dropped_no_motif_all,
        "dropped_too_short": dropped_too_short_all,
        "pct_kept_after_trim_of_raw": round(pct_kept, 6),
        "pct_mapped_of_total_reads": round(pct_mapped_trimmed, 6),
        "pct_mapped_of_raw_reads": round(pct_mapped_raw, 6),
        "pct_of_all_mapped_reads": 100.0 if mapped_reads_all else 0.0
    }]

    for contig, length in contig_lengths.items():
        m = contig_mapped_all.get(contig, 0)
        if (not INCLUDE_ZEROS) and m == 0:
            continue
        pct_total_trimmed = (m / total_reads_all * 100.0) if total_reads_all else 0.0
        pct_total_raw = (m / raw_reads_all * 100.0) if raw_reads_all else 0.0
        pct_mapped_reads = (m / mapped_reads_all * 100.0) if mapped_reads_all else 0.0
        rows.append({
            "contig": contig,
            "length_bp": length,
            "mapped_reads_primary": m,
            "total_reads_primary": total_reads_all,
            "raw_reads_r2": raw_reads_all,
            "dropped_no_motif": "",
            "dropped_too_short": "",
            "pct_kept_after_trim_of_raw": "",
            "pct_mapped_of_total_reads": round(pct_total_trimmed, 6),
            "pct_mapped_of_raw_reads": round(pct_total_raw, 6),
            "pct_of_all_mapped_reads": round(pct_mapped_reads, 6),
        })

    df = pd.DataFrame(rows)

    # Sort contigs by mapped reads, keep __ALL__ on top
    if len(df) > 1:
        df_contigs = df[df["contig"] != "__ALL__"].sort_values("mapped_reads_primary", ascending=False)
        df_all = df[df["contig"] == "__ALL__"]
        df = pd.concat([df_all, df_contigs], ignore_index=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".xlsx":
        df.to_excel(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)

    print(f"[done] Wrote: {out_path}", file=sys.stderr)

    if not KEEP_BAMS:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
