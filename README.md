# BRACE

Official implementation of **BRACE** (Bipolar Reference-Aware Calibration) for multimodal
sentiment analysis with missing modalities, on CMU-MOSI and CMU-MOSEI.

BRACE encodes every observed modality into slots, builds a quality-weighted state token and
fuses it with the slots in a joint transformer. The fused proxy is then calibrated against a
memory bank of training proxies: higher- and lower-label reference centroids are retrieved
around the current prediction and drive a gated signed update of the proxy, which is scored
by the same prediction head.

## Protocol

Fixed missing modalities: each run observes one fixed modality subset in the train, validation
and test splits alike, and the remaining modalities are zeroed out. The seven subsets are
`l`, `v`, `a`, `lv`, `la`, `va` and `lva` (l = text, a = audio, v = vision).

## Setup

```bash
pip install -r requirements.txt
```

A CUDA GPU is required. All runs use seed 42 and deterministic kernels, so repeated runs of
the same configuration give identical results.

## Data

Training expects `data/mosi.pkl` and `data/mosei.pkl`. Each file is a dict with `train`,
`valid` and `test` splits; every split holds `text` (N×1×1024), `audio` (N×1×512),
`vision` (N×1×1024) features and `regression_labels` (N×1) in [-3, 3].

| Dataset | train | valid | test |
|---|---|---|---|
| CMU-MOSI | 1284 | 229 | 686 |
| CMU-MOSEI | 16326 | 1871 | 4659 |

`scripts/prepare_data.py` builds these files from the pre-extracted feature pickles
`CMUMOSI_new.pkl` / `CMUMOSEI_new.pkl` without changing features, labels or the official splits:

```bash
python scripts/prepare_data.py --input /path/to/CMUMOSI_new.pkl --output data/mosi.pkl
python scripts/prepare_data.py --input /path/to/CMUMOSEI_new.pkl --output data/mosei.pkl
```

## Training

```bash
python train.py --dataset mosei --subset lva --output runs/mosei/lva
```

`--dataset` is `mosi` or `mosei`, `--subset` is one of the seven subsets above.
Hyperparameters of every subset are in `configs/mosi.yaml` and `configs/mosei.yaml`.
Training logs go to `<output>/metrics.jsonl`.

## Layout

```text
train.py                        command-line entry: training and per-epoch evaluation
configs/                        hyperparameters for CMU-MOSI and CMU-MOSEI
brace/
  model.py                      BRACE: anchor encoder + reference calibration, training objective
  modules/
    slots.py                    per-modality slot projection
    transformer.py              masked joint transformer block
    encoder.py                  Stage 1: quality-consistent anchor (SlotFusionEncoder)
    memory.py                   Stage 2: global state-label memory (MemoryBank)
    calibration.py              Stage 3: bipolar reference-aware calibration (RetrievalEditor)
  data.py                       data loading and fixed-subset masking
  engine.py                     one training epoch and evaluation
  metrics.py                    Acc-2 / F1 / Acc-7 / MAE
  utils.py                      seeding, config loading, learning-rate schedule
  constants.py                  modalities, subsets and shared constants
scripts/prepare_data.py         builds data/*.pkl from the feature pickles
```

## License

MIT, see [LICENSE](LICENSE).
