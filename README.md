# RADAR: Acquisition-Invariant Attention Pooling over a Frozen Chest-Radiograph Foundation Model for Tuberculosis Screening

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-adidukre%2Fradar--tb-yellow)](https://huggingface.co/adidukre/radar-tb)

Official code for **RADAR**, our entry to MICCAI 2026 TREAT-MMTB Task 2 (tuberculosis vs. normal
chest radiograph classification).

RADAR keeps the chest-radiograph foundation model [RAD-DINO](https://huggingface.co/microsoft/rad-dino)
frozen and trains only small cross-attention read-out heads. A learned tuberculosis query pools
patch tokens from four encoder depths, so the classifier can attend to localized disease instead of
averaging the whole image. Heads trained with a modality adversary remove acquisition-specific
information, and the final prediction ensembles seeds, lung-cropped and full-frame views, and two
input resolutions.

## Method

1. **Preprocessing.** Any bit depth or format (PNG, JPEG, TIFF, DICOM) is windowed to 8-bit
   grayscale. Lungs are located with the TorchXRayVision ChestX-Det PSPNet and cropped; an
   implausible mask falls back to the full frame. Inverted (MONOCHROME1-style) exports are detected
   against a reference embedding and corrected.
2. **Frozen encoder.** RAD-DINO ViT-B/14 at 518 px gives a 37 x 37 token grid. Tokens from blocks
   {3, 6, 9, 12} are concatenated into 3072-d descriptors.
3. **Attention read-out.** Each head is a single-query GLoRI / ML-Decoder cross-attention decoder
   trained with AdamW and a weight EMA. Three head groups share one encoder pass:

   | group | heads | training |
   |---|---|---|
   | `dep` | 6 | CR-modality slice, 3 plain seeds + 3 seeds with style randomization (RandConv, Fourier amplitude, histogram matching) and weak/strong consistency |
   | `adv` | 3 | all modalities, gradient-reversal modality adversary (lambda = 1) |
   | `cdan` | 3 | all modalities, prediction-conditioned (CDAN+E) modality adversary |

4. **High-resolution read-out.** The same frozen encoder at 700 px (50 x 50 tokens) is read by
   three further adversarial heads, Platt-mapped onto the 518 px probability scale.
5. **Decision rule.**

   ```
   p_518 = sum_k w_k [0.90 p_k(crop) + 0.10 p_k(full)],   w = (dep 0.39, adv 0.26, cdan 0.35)
   p_700 = sigmoid(1.425 * logit(p_hires) + 0.057)
   p     = 0.85 p_518 + 0.15 p_700,                     TB if p >= 0.94
   ```

Acquisition metadata is used only as the adversary target during training. Inference is
image-only, offline and deterministic.

## Results

| setting | metric | RADAR |
|---|---|---|
| TREAT-MMTB Task 2, external cohort (Korea, Mongolia, Peru, Philippines) | F1 | 0.8375 |
| TREAT-MMTB Task 2, internal validation (1940 images), deployed container | F1 | 0.9840 |
| Five held-out public sites, four countries | worst-site F1 | 0.9064 (vs. 0.8815 for the `dep` group alone) |

## Installation

```bash
git clone https://github.com/adinathdukre/RADAR-TB.git
cd RADAR-TB
conda create -n radar python=3.10 -y && conda activate radar
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
```

## Pretrained weights

The 15 trained heads and the polarity reference are hosted at
[adidukre/radar-tb](https://huggingface.co/adidukre/radar-tb):

```bash
hf download adidukre/radar-tb --local-dir weights
```

Alternatively, the step below downloads the heads together with RAD-DINO and the lung segmenter:

```bash
python scripts/prepare.py --weights-repo adidukre/radar-tb
```

```
weights/
  dep_plain_seed{42,1337,2024}.pt
  dep_aug_seed{42,1337,2024}.pt
  adv_seed{42,1337,2024}.pt
  cdan_seed{42,1337,2024}.pt
  hires_seed{42,1337,2024}.pt
  polarity_ref.npy
```

## Inference

```bash
python scripts/predict.py --input /path/to/images --output /path/to/output
```

This writes `/path/to/output/prediction.csv` with columns `filename,TB/Normal`, one row per image
found under the input directory (searched recursively). Add `--with-prob` to include the
probability and `--threshold` to change the operating point.

From Python:

```python
from radar.inference import Radar

model = Radar()
probs, labels = model.predict(["/path/to/image1.png", "/path/to/image2.png"])
```

### Docker

The container reads `/input` and writes `/output/prediction.csv`, runs with `--network none`, and
falls back to CPU when no GPU is visible.

```bash
HF_HOME=checkpoints/hf RADAR_CACHE=checkpoints python scripts/prepare.py --weights-repo adidukre/radar-tb
docker build -f docker/Dockerfile -t radar-tb .
docker run --rm --gpus all --network none \
    -v /path/to/images:/input:ro \
    -v /path/to/output:/output \
    radar-tb
```

## Training

Paths are set through environment variables (defaults in parentheses):

| variable | meaning |
|---|---|
| `RADAR_DATA` | challenge data root holding `train/`, `test/`, `train.csv`, `test.csv` (`data/Task2/Data`) |
| `RADAR_WORK` | lung boxes, token caches and public datasets (`work`) |
| `RADAR_TOKENS` | token cache, needs ~65 GB at 518 px and ~111 GB at 700 px (`work/tokens`) |
| `RADAR_WEIGHTS` | trained heads (`weights`) |
| `RADAR_CACHE` | segmenter weights (`checkpoints`) |

```bash
export RADAR_DATA=/path/to/Task2/Data
export RADAR_WORK=/path/to/work
```

Montgomery and Shenzhen (NLM) are downloaded automatically and used for checkpoint selection
only. The high-resolution heads additionally select on the Pakistan (Mendeley) and India (NITRD DA/DB)
sets; place them under
`$RADAR_WORK/external/{pakistan,india}/` either as `tb/` and `normal/` folders or with a
`labels.csv` (`filename,label`).

```bash
# 1. backbone, segmenter and selection sets
python scripts/prepare.py --proxies montgomery shenzhen

# 2. lung boxes
python scripts/segment.py --splits train test montgomery shenzhen pakistan india

# 3. dep group: image-space training through the frozen encoder
for s in 42 1337 2024; do
  python scripts/train_base.py --seed $s
  python scripts/train_base.py --seed $s --aug
done

# 4. adv and cdan groups: trained on cached 518 px tokens
python scripts/cache_tokens.py --res 518 --splits train test montgomery shenzhen
python scripts/train_adversarial.py --method dann --lam 1.0
python scripts/train_adversarial.py --method cdan --lam 1.0

# 5. high-resolution group and its Platt map
python scripts/cache_tokens.py --res 700 --splits train test montgomery shenzhen pakistan india
python scripts/train_adversarial.py --method dann --lam 1.0 --res 700
python scripts/fit_platt.py

# 6. polarity reference
python scripts/make_polarity_ref.py
```

Blend weights, Platt parameters and the threshold live in `radar/config.py`. Checkpoints are
selected on public held-out sites and never on the challenge test data; the Platt map is fitted on
internal validation only.

## Repository structure

```
radar/
  config.py          paths and deployed constants
  preprocessing.py   image decoding, windowing, crop and square padding
  segmentation.py    PSPNet lung boxes
  backbone.py        frozen RAD-DINO wrapper
  glori.py           cross-attention read-out head
  heads.py           head construction, loading, EMA, gradient reversal
  augment.py         style randomization
  tokens.py          token caching
  external.py        public selection datasets
  metrics.py         F1, AUROC, worst-site selection
  inference.py       end-to-end predictor
scripts/             data preparation, training and inference entry points
docker/              offline inference container
```

## Citation

```bibtex
@inproceedings{rege2026radar,
  title     = {RADAR: Acquisition-Invariant Attention Pooling over a Frozen Chest-Radiograph Foundation Model for Tuberculosis Screening},
  author    = {Rege, Atharva Atul and Dukre, Adinath Madhavrao and Shah, Sarth Santosh and Razzak, Imran},
  booktitle = {MICCAI 2026 TREAT-MMTB Challenge},
  year      = {2026}
}
```

## Acknowledgements

RADAR builds on [RAD-DINO](https://huggingface.co/microsoft/rad-dino),
[TorchXRayVision](https://github.com/mlmed/torchxrayvision) and the
[ML-Decoder](https://github.com/Alibaba-MIIL/ML_Decoder) / GLoRI read-out. Please follow their
licenses when using the pretrained models.

We thank the organizers of the [MICCAI 2026 TREAT-MMTB challenge](https://treat-mmtb.mi2rl.co/)
for the Task 2 training and evaluation data, and the providers of the public chest radiograph
datasets used for model selection and evaluation:

- **Montgomery County and Shenzhen** (U.S. National Library of Medicine): S. Jaeger et al.,
  "Two public chest X-ray datasets for computer-aided screening of pulmonary diseases,"
  *Quantitative Imaging in Medicine and Surgery*, 2014.
- **TBX11K**: Y. Liu et al., "Rethinking computer-aided tuberculosis diagnosis," *CVPR*, 2020.
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (non-commercial).
- **Mendeley**: S. Kiran and I. Jabeen, "Dataset of Tuberculosis Chest X-rays Images," Mendeley
  Data, [doi:10.17632/8j2g3csprk.2](https://doi.org/10.17632/8j2g3csprk.2).
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **India DA and DB** (National Institute of Tuberculosis and Respiratory Diseases, New Delhi):
  A. Chauhan et al., "Role of Gist and PHOG features in computer-aided diagnosis of tuberculosis
  without segmentation," *PLoS ONE*, 2014.

These datasets were used only for checkpoint selection and evaluation. No exter
contributes training gradients to the released weights. Each dataset remains subject to its own
terms; please obtain it from the original source.
