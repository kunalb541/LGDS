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

# Run the CA package
python lgds.py

# Compile all three manuscripts with aux files in outputs/logs:
#   paper.tex          — short LGDS note, non-blind (RevTeX/PRE format)
#   paper_blind.tex    — short LGDS note, blind (elsarticle/SCL format)
#   paper_expanded.tex — main formal paper, non-blind (elsarticle format)
for src in paper.tex paper_blind.tex paper_expanded.tex; do
  latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=outputs/logs "$src"
  cp "outputs/logs/${src%.tex}.pdf" "./${src%.tex}.pdf"
done

echo
echo "Done."
echo "Top level now contains:"
ls -1
echo
echo "Figures found:"
ls -1 outputs/figures 2>/dev/null || true