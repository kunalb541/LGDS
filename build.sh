#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

mkdir -p outputs/data outputs/figures outputs/tables outputs/logs

# Move common LaTeX auxiliary files into logs if they exist
shopt -s nullglob
for f in *.aux *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz *.bbl *.blg *.nav *.snm; do
  mv "$f" outputs/logs/
done
shopt -u nullglob

# Generate numerical macros from the LGDS computation
python lgds.py

# Compile the manuscript with aux files in outputs/logs
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=outputs/logs paper.tex

# Copy final PDF back to top level
cp outputs/logs/paper.pdf ./paper.pdf

echo
echo "Done."
echo "Top level now contains:"
ls -1