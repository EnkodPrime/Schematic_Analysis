# The Mathematics Behind ML/AI — A Weighted Map

A single self-contained HTML page: an interactive, bilingual (EN / БГ) map of the
mathematics that actually carries weight in machine learning.

**Live page:** https://enkodprime.github.io/Schematic_Analysis/ *(after GitHub Pages is enabled — see below)*

## What is in it

- **88 mathematical topics**, each weighted 1–10 by how often it shows up in real
  ML/AI work — reading papers, reading model code, debugging a training run.
- **A radial map** of eight branches (linear algebra, calculus, optimization,
  probability, statistics, information theory, discrete math, geometry &
  numerics). Radius encodes weight: 10 at the centre, 1 at the rim. The thin arcs
  are prerequisites.
- **Per topic:** the formula, a symbol-by-symbol table, a worked numeric example,
  an illustration, and an ego graph of the ML/AI concepts that use it.
- **70 ML/AI concepts** across 8 groups, each clickable in reverse: pick a
  concept and see the mathematics it needs, ordered by how central that maths is.
- **A 64-entry symbol glossary** — click any sign for how it is read, what it
  means, an example and a diagram.
- **EN / БГ language switch**, light and dark themes, keyboard-accessible.

No build step, no dependencies, no network calls: `index.html` is the whole site.

## Enabling GitHub Pages

Settings → Pages → *Build and deployment* → Source: **Deploy from a branch** →
Branch: **main**, folder: **/docs** → Save. The URL above goes live in ~1 minute.

`.nojekyll` is present so Jekyll does not touch the file.

## Local viewing

Open `docs/index.html` in any browser — `file://` works, nothing else needed.
