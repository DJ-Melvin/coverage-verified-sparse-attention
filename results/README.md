# Figure inputs

This directory holds the files the paper's figures are drawn from;
`bash scripts/make_figures.sh` redraws all of them on a CPU.
[REPRODUCIBILITY.md](../REPRODUCIBILITY.md) lists which figure reads which
file.

The 23 JSON files shipped here are the plotting inputs. `.gitignore` lists
them explicitly and ignores new experiment outputs and raw traces by default.
Redrawing figures uses these stored results; rerunning the experiments is a
separate step described in the reproducibility guide.

The launchers in `scripts/` write their outputs here as well: experiments of
the submitted version to `results/`, experiments added in the camera-ready
version to `results/camera_ready/`.

Several files are compact summaries of raw diagnostic traces
(`*_summary.json`). The plotting scripts rebuild a summary from the raw trace
when the summary file is absent.
