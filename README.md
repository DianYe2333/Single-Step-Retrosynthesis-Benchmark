# Single-Step Retrosynthesis Benchmark

AIDD8122 course project comparing NeuralSym-style template ranking and RetroSim-style precedent retrieval on a shared USPTO-50K split. Both methods use RDChiral templates extracted exclusively from training reactions. The implementations are adaptations of the cited methods.

## Reference run

CPU run on 500 test records, seed 42, 30 training epochs, reaction class withheld. Accuracy requires an exact match of canonicalized precursor sets, retaining stereochemistry.

| Method | Top-1 | Top-5 | Top-10 | Median CPU time |
| --- | ---: | ---: | ---: | ---: |
| NeuralSym-style | 38.8% | 62.6% | 68.6% | 36 ms |
| RetroSim-style | 38.4% | 67.2% | 74.2% | 129 ms |

The original metrics, figures, test indices, data hashes and split overlap audit are in [reference](reference/). Timing includes ranking and template application and depends on hardware and cache state. The split contains duplicate reactions and repeated products across train and test, so the benchmark also reports metrics after excluding each form of overlap.

## Setup and run

Use Python 3.12. Run commands from the repository root. Activate a virtual environment with `source .venv/bin/activate` on Linux/WSL or `.venv\Scripts\Activate.ps1` in PowerShell after creating it.

```text
python -m venv .venv
python -m pip install -r requirements.txt
python fetch_data.py
python prepare_data.py --workers 4
python benchmark.py --samples 500 --epochs 30
python analyze.py
```

The full run trains on the complete 40,008-record training split on CPU. The validation split is retained for provenance and was not used for model selection. Downloads are pinned to an OpenRetro commit and checked with SHA-256. Generated predictions, plots, error cases, timing, model weights and metrics go to `runs/benchmark/`. Use a separate `--output-dir` for each run to retain earlier results. `requirements-lock.txt` records the complete original Windows experiment environment.

For a small pipeline check, use distinct directories so the full dataset and run remain available:

```text
python prepare_data.py --limit 100 --workers 2 --prepared-dir data/smoke
python benchmark.py --prepared-dir data/smoke --samples 10 --epochs 1 --output-dir runs/smoke
python analyze.py --output-dir runs/smoke
```

Smoke metrics are pipeline diagnostics. Raw data, generated runs, model weights and local document builders are excluded from Git. The upstream sources provide their own data and code terms. This repository preserves the upstream MIT notices for the OpenRetro and RetroSim code. No new license grant is declared for the project-authored files.

## Sources

- [Segler and Waller (2017), Neural-Symbolic Machine Learning for Retrosynthesis and Reaction Prediction](https://doi.org/10.1002/chem.201605499)
- [Coley et al. (2017), Computer-Assisted Retrosynthesis Based on Molecular Similarity](https://doi.org/10.1021/acscentsci.7b00355)
- [OpenRetro, pinned data and neural network implementation](https://github.com/coleygroup/openretro/tree/96fb40cd622ae4a7645e51b50479c930341afee8)
- [RetroSim, similarity scoring reference](https://github.com/connorcoley/retrosim/tree/0a272f0b5de833c448f41491e81e4dc00b4d85b0)
- [RDChiral, template extraction and application](https://github.com/connorcoley/rdchiral)
