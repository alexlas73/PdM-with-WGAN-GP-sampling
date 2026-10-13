# Changelog

## 2.1.0 (13 October 2026)
* Adds `results/round2_defaults/`: both generative strategies rerun on the 30 splits with pre-specified generator
  settings (tabgan 2.2.2 defaults), Table 8 of the paper (editor's request, second revision round).
* `ftaa.experiment --generator-settings defaults` and `ftaa.analysis --replace` to reproduce it;
  `TABGAN_DEFAULTS` in `ftaa/config.py`. Results of the main archive are unchanged.
  
## 2.0.0 (5 October 2026)
* Replaces the earlier single-partition analysis (version 1, release tag `v1-earlier-analysis`).
* Repeated-split evaluation of eight strategies (30 stratified 70/15/15 splits), as reported in the IJIKM article.
* Archived predictions (240 runs), all result tables and Figures 5-10.
* Pipeline packaged as `ftaa` (Stage 1 generator search, Stage 2 experiment, Stage 3 analysis, t-SNE figures),
  with a Colab notebook, tests, and continuous integration.
