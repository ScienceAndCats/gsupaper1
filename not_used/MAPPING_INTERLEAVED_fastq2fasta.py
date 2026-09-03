#!/usr/bin/env python3
"""
PyCharm-run version: edit USER SETTINGS below and click Run.

This version (interleaved mode):
- Assumes FASTQs are INTERLEAVED paired-end:
    R1 (58 bp, barcodes) then R2 (17 bp, cDNA) for each read pair.
- Uses ONLY R2 reads from each interleaved pair.
- Optionally trims trailing Ns from R2 before alignment (toggle).
- Aligns reads to REF_FASTA with bwa mem (single-end)
- Outputs one CSV or XLSX:
    - __ALL__ summary row: overall percent mapped + simple read counts
    - per-contig mapped read counts and percentages

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

FASTQ_DIR = "/home/hanst/PhETRIseq/original"

# Combined reference FASTA (all genomes/contigs in one file)
REF_FASTA = "/home/hanst/PhETRIseq/original/U00096_CP000255.fasta"

# Output file (set .csv OR .xlsx). Single output.
OUT_FILE = "/home/hanst/PhETRIseq/original/mapping_stats3.csv"

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
# R2 TRIMMING / N-HANDLING
# ----------------------------

# If True, trim trailing Ns from R2 before alignment.
# If False, use R2 exactly as in the FASTQ.
TRIM_R2_TRAILING_NS = True

# Minimum length required after trimming; shorter reads are discarded
MIN_LEN_AFTER_TRIM = 10

# ----------------------------
# LEGACY MOTIF SETTINGS (unused here)
# ----------------------------
# Kept around in case you want to switch back to motif-based trimming later.

# Motif marking end of construct/barcode region, immediately before the random hexamer
MOTIF = "CAGAGAA"

# Remove the random hexamer immediately after the motif?
REMOVE_HEXAMER = True
HEX_LEN = 6

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


def open_maybe_gz(path: Path, mode="rt"):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode)


def safe_tag_from_fastq(fq: Path):
    tag = fq.name
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
    Legacy trimming function (currently unused).
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


def align_interleaved_r2_to_bam(fq_path: Path, ref_fa: Path, bam_path: Path,
                                bwa_exe: str, samtools_exe: str):
    """
    Assume FASTQ is INTERLEAVED paired-end:
      R1 (barcodes) then R2 (cDNA) for each pair.

    We:
      - Read 8-line blocks: 4 lines R1, 4 lines R2
      - Ignore R1 entirely
      - For R2:
          * If TRIM_R2_TRAILING_NS is True:
              - Strip trailing Ns (and matching qualities)
              - Drop reads shorter than MIN_LEN_AFTER_TRIM
          * Else:
              - Use R2 exactly as-is
      - Stream kept R2 reads to bwa mem as single-end

    Returns stats dict.
    """
    stats = {
        "raw_reads_r2": 0,              # R2 reads examined
        "kept_reads_after_trim": 0,     # R2 reads actually sent to bwa
        "dropped_no_motif": 0,          # unused in this mode
        "dropped_too_short": 0,         # used if trimming makes read too short
        "kept_untrimmed_no_motif": 0,   # unused in this mode
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

    try:
        with open_maybe_gz(fq_path, "rt") as fin:
            while True:
                # R1 (ignored)
                r1_header = fin.readline()
                if not r1_header:
                    break  # EOF
                r1_seq = fin.readline()
                r1_plus = fin.readline()
                r1_qual = fin.readline()

                # R2 (used)
                r2_header = fin.readline()
                if not r2_header:
                    break  # incomplete pair at EOF
                r2_seq = fin.readline().rstrip("\n")
                r2_plus = fin.readline()
                r2_qual = fin.readline().rstrip("\n")

                stats["raw_reads_r2"] += 1

                if TRIM_R2_TRAILING_NS:
                    # Trim trailing Ns (case-insensitive)
                    seq_u = r2_seq.upper()
                    trimmed_len = len(seq_u.rstrip("N"))
                    if trimmed_len < MIN_LEN_AFTER_TRIM:
                        stats["dropped_too_short"] += 1
                        continue
                    trimmed_seq = r2_seq[:trimmed_len]
                    trimmed_qual = r2_qual[:trimmed_len]
                    seq_to_write = trimmed_seq
                    qual_to_write = trimmed_qual
                else:
                    seq_to_write = r2_seq
                    qual_to_write = r2_qual

                stats["kept_reads_after_trim"] += 1

                # Write R2 FASTQ record (possibly trimmed) to bwa stdin
                bwa.stdin.write(r2_header.encode())
                bwa.stdin.write((seq_to_write + "\n").encode())
                bwa.stdin.write(r2_plus.encode())
                bwa.stdin.write((qual_to_write + "\n").encode())

    finally:
        if bwa.stdin:
            bwa.stdin.close()

    # Collect stderr/returncodes
    sort_out, sort_err = sort.communicate()
    view_out, view_err = view.communicate()
    bwa_out, bwa_err = bwa.communicate()

    if bwa.returncode != 0 or view.returncode != 0 or sort.returncode != 0:
        msg = [
            f"ERROR: pipeline failed for {fq_path}",
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

    # Use all FASTQs (interleaved), except ones matching SKIP_IF_NAME_CONTAINS
    fq_files = []
    for p in fastqs:
        nlow = p.name.lower()
        if any(s in nlow for s in SKIP_IF_NAME_CONTAINS):
            continue
        fq_files.append(p)

    if not fq_files:
        raise SystemExit("ERROR: No FASTQ files to process after applying SKIP_IF_NAME_CONTAINS filter.")

    if TMP_BAM_DIR:
        tmp_dir = Path(TMP_BAM_DIR).expanduser().resolve()
    else:
        tmp_dir = out_path.parent / (out_path.stem + ".tmp_bams")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Aggregates
    total_reads_all = 0              # total reads entering alignment (kept R2)
    mapped_reads_all = 0             # mapped reads in BAM
    contig_lengths = {}
    contig_mapped_all = {}

    raw_reads_all = 0
    dropped_no_motif_all = 0         # unused in this mode
    dropped_too_short_all = 0        # from MIN_LEN_AFTER_TRIM
    kept_untrimmed_no_motif_all = 0  # unused in this mode

    print(f"[info] FASTQ_DIR: {fastq_dir}", file=sys.stderr)
    trim_mode = "trim Ns" if TRIM_R2_TRAILING_NS else "R2 as-is"
    print(f"[info] Found {len(fq_files)} interleaved FASTQ file(s) to process ({trim_mode}).", file=sys.stderr)

    for i, fq in enumerate(fq_files, start=1):
        tag = safe_tag_from_fastq(fq)
        bam = tmp_dir / f"{tag}.primary.bam"

        print(f"[info] ({i}/{len(fq_files)}) Mapping interleaved FASTQ ({trim_mode}): {fq.name}", file=sys.stderr)

        trim_stats = align_interleaved_r2_to_bam(fq, ref_fa, bam, bwa, samtools)

        raw_reads_all += trim_stats["raw_reads_r2"]
        total_reads_all += trim_stats["kept_reads_after_trim"]
        dropped_no_motif_all += trim_stats["dropped_no_motif"]
        dropped_too_short_all += trim_stats["dropped_too_short"]
        kept_untrimmed_no_motif_all += trim_stats["kept_untrimmed_no_motif"]

        # Count mapped/unmapped from BAM
        total_reads_bam = int(subprocess.check_output([samtools, "view", "-c", str(bam)]).decode().strip())
        mapped_reads = int(subprocess.check_output([samtools, "view", "-c", "-F", "4", str(bam)]).decode().strip())
        mapped_reads_all += mapped_reads

        # Sanity check
        if total_reads_bam != trim_stats["kept_reads_after_trim"]:
            print(
                f"[warn] BAM record count ({total_reads_bam}) != kept_reads_after_trim "
                f"({trim_stats['kept_reads_after_trim']}) for {fq.name}",
                file=sys.stderr
            )

        idx = subprocess.check_output([samtools, "idxstats", str(bam)]).decode()
        lens, mapped_by_contig = parse_idxstats(idx)

        if not contig_lengths:
            contig_lengths = lens

        for contig, mcount in mapped_by_contig.items():
            contig_mapped_all[contig] = contig_mapped_all.get(contig, 0) + mcount

    pct_mapped_trimmed = (mapped_reads_all / total_reads_all * 100.0) if total_reads_all else 0.0
    pct_mapped_raw = (mapped_reads_all / raw_reads_all * 100.0) if raw_reads_all else 0.0
    pct_kept = (total_reads_all / raw_reads_all * 100.0) if raw_reads_all else 0.0

    # Output rows
    rows = [{
        "contig": "__ALL__",
        "length_bp": "",
        "mapped_reads_primary": mapped_reads_all,
        "total_reads_primary": total_reads_all,   # R2 reads entering alignment
        "raw_reads_r2": raw_reads_all,
        "dropped_no_motif": dropped_no_motif_all,
        "dropped_too_short": dropped_too_short_all,
        "kept_untrimmed_no_motif": kept_untrimmed_no_motif_all,
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
            "kept_untrimmed_no_motif": "",
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
