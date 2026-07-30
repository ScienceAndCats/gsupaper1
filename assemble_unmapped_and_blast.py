#!/usr/bin/env python3
"""
Assemble unmapped R2 reads and BLAST resulting contigs.

Workflow:
  1. Take a single FASTQ(/gz) of unmapped, trimmed R2 reads (from your mapping script).
  2. Assemble reads with MEGAHIT (single-end).
  3. BLAST assembled contigs with blastn against a user-specified database.
  4. Write tab-delimited BLAST hits (outfmt 6).

Requirements (conda examples):
  conda install -c bioconda -c conda-forge megahit blast pandas
"""

import shutil
import subprocess
import sys
from pathlib import Path
import pandas as pd

# ==================================
# USER SETTINGS (edit in PyCharm)
# ==================================

# Unmapped reads FASTQ from your previous script
UNMAPPED_FASTQ = "/home/hanst/PhETRIseq/jrg07PMP/JRG07-Sample-P3/R2_unmapped_trimmed.fastq"

# Output directory for assembly + BLAST results
OUTPUT_DIR = "/home/hanst/PhETRIseq/jrg07PMP/JRG07-Sample-P3/unmapped_assembly"

# If OUTPUT_DIR already exists:
#   True  -> delete it and rerun MEGAHIT
#   False -> if final.contigs.fa exists, reuse it; otherwise error
OVERWRITE_ASSEMBLY = True

# MEGAHIT settings
MEGAHIT_MIN_CONTIG_LEN = 200
THREADS = 16

# BLAST settings
# Path to local BLAST database *prefix* (e.g. "/path/to/db/nt" if you have nt locally),
# or your own custom db built with makeblastdb.
BLAST_DB = "/path/to/blast_db_prefix"

# BLAST output file (tab-delimited)
BLAST_OUT_TSV = "/home/hanst/PhETRIseq/jrg07PMP/JRG07-Sample-P3/unmapped_assembly/contigs_vs_db.blastn.tsv"

# Maximum target sequences per contig
BLAST_MAX_TARGET_SEQS = 10

# E-value cutoff
BLAST_EVALUE = 1e-10

# ==================================
# END USER SETTINGS
# ==================================


def which_or_die(exe: str) -> str:
    path = shutil.which(exe)
    if not path:
        raise SystemExit(
            f"ERROR: '{exe}' not found in PATH.\n"
            f"Install with conda, for example:\n"
            f"  conda install -c bioconda -c conda-forge {exe}\n"
        )
    return path


def run(cmd, cwd=None):
    """Run a command, raising on error."""
    print(f"[cmd] {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True, cwd=cwd)


def run_megahit(unmapped_fastq: Path, out_dir: Path,
                threads: int, min_contig_len: int,
                overwrite: bool) -> Path:
    """
    Run MEGAHIT in single-end mode on the unmapped reads FASTQ.
    Handles existing OUTPUT_DIR depending on `overwrite`.
    """
    megahit_exe = which_or_die("megahit")

    contigs = out_dir / "final.contigs.fa"

    if out_dir.exists():
        if overwrite:
            print(f"[info] OUTPUT_DIR exists, removing for fresh assembly: {out_dir}", file=sys.stderr)
            shutil.rmtree(out_dir)
        else:
            if contigs.exists():
                print(
                    f"[info] OUTPUT_DIR exists and final.contigs.fa found; "
                    f"reusing existing assembly.",
                    file=sys.stderr,
                )
                return contigs
            else:
                raise SystemExit(
                    f"ERROR: OUTPUT_DIR {out_dir} already exists and no final.contigs.fa found.\n"
                    f"Either delete it manually or set OVERWRITE_ASSEMBLY = True."
                )

    # Ensure parent exists, but do NOT create out_dir itself;
    # MEGAHIT insists that -o does not exist when it starts.
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        megahit_exe,
        "-r", str(unmapped_fastq),
        "-o", str(out_dir),
        "-t", str(threads),
        "--min-contig-len", str(min_contig_len),
    ]

    run(cmd)

    if not contigs.exists():
        raise SystemExit(f"ERROR: MEGAHIT did not produce final.contigs.fa in {out_dir}")
    print(f"[info] MEGAHIT contigs: {contigs}", file=sys.stderr)
    return contigs


def run_blastn(contigs_fa: Path, blast_db: str, blast_out: Path,
               threads: int, max_target_seqs: int, evalue: float):
    """
    Run blastn on assembled contigs vs the specified database.
    Output in tabular format (outfmt 6).
    """
    blastn_exe = which_or_die("blastn")

    blast_out.parent.mkdir(parents=True, exist_ok=True)

    outfmt = "6 std qlen slen"  # standard cols + query length and subject length

    cmd = [
        blastn_exe,
        "-query", str(contigs_fa),
        "-db", blast_db,
        "-out", str(blast_out),
        "-outfmt", outfmt,
        "-num_threads", str(threads),
        "-max_target_seqs", str(max_target_seqs),
        "-evalue", str(evalue),
    ]

    run(cmd)
    if not blast_out.exists():
        raise SystemExit(f"ERROR: BLAST output not created: {blast_out}")

    print(f"[info] BLAST results written to: {blast_out}", file=sys.stderr)


def summarize_blast(blast_out: Path):
    """
    Optional: read BLAST TSV and print a quick summary.
    """
    if not blast_out.exists():
        print("[warn] No BLAST output to summarize.", file=sys.stderr)
        return

    cols = [
        "qseqid", "sseqid", "pident", "length", "mismatch", "gapopen",
        "qstart", "qend", "sstart", "send", "evalue", "bitscore",
        "qlen", "slen"
    ]
    try:
        df = pd.read_csv(blast_out, sep="\t", header=None, names=cols)
    except Exception as e:
        print(f"[warn] Could not parse BLAST TSV for summary: {e}", file=sys.stderr)
        return

    print(f"[summary] Total BLAST hits: {len(df)}", file=sys.stderr)
    if len(df) == 0:
        return

    # Quick per-contig hit count
    hits_per_contig = df["qseqid"].value_counts()
    top = hits_per_contig.head(10)
    print("[summary] Top contigs by number of hits:", file=sys.stderr)
    for qid, count in top.items():
        print(f"  {qid}: {count} hits", file=sys.stderr)


def main():
    unmapped_fastq = Path(UNMAPPED_FASTQ).expanduser().resolve()
    out_dir = Path(OUTPUT_DIR).expanduser().resolve()
    blast_out = Path(BLAST_OUT_TSV).expanduser().resolve()

    if not unmapped_fastq.exists():
        raise SystemExit(f"ERROR: UNMAPPED_FASTQ not found: {unmapped_fastq}")

    print(f"[info] Unmapped FASTQ: {unmapped_fastq}", file=sys.stderr)
    print(f"[info] Output directory: {out_dir}", file=sys.stderr)

    # 1) Assemble with MEGAHIT
    contigs_fa = run_megahit(
        unmapped_fastq=unmapped_fastq,
        out_dir=out_dir,
        threads=THREADS,
        min_contig_len=MEGAHIT_MIN_CONTIG_LEN,
        overwrite=OVERWRITE_ASSEMBLY,
    )

    # 2) BLAST contigs
    if BLAST_DB and BLAST_DB != "/path/to/blast_db_prefix":
        run_blastn(
            contigs_fa=contigs_fa,
            blast_db=BLAST_DB,
            blast_out=blast_out,
            threads=THREADS,
            max_target_seqs=BLAST_MAX_TARGET_SEQS,
            evalue=BLAST_EVALUE,
        )
        summarize_blast(blast_out)
    else:
        print(
            "[warn] BLAST_DB is not set to a real database path. "
            "Assembly completed, but BLAST was skipped.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
