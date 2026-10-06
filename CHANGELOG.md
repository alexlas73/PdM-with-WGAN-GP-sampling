# Changelog

## 2.0.0 (October 2026)
* Replaces the earlier single-partition analysis (version 1, release tag `v1-earlier-analysis`).
* Repeated-split evaluation of eight strategies (30 stratified 70/15/15 splits), as reported in the IJIKM article.
* Archived predictions (240 runs), all result tables and Figures 5-10.
* Pipeline packaged as `ftaa` (Stage 1 generator search, Stage 2 experiment, Stage 3 analysis, t-SNE figures),
  with a Colab notebook, tests, and continuous integration.
