# Sign Language LLM

## Project idea

This project extends prior work on multilingual semantic representations in mBERT from variation across spoken/written languages to variation across modalities.

The central question is:

> Can a model trained on sign-language input develop semantic representations that align with spoken-language semantic spaces, beyond modality-specific visuomotor form?

## Working hypothesis

A sign-language encoder may exhibit a representational hierarchy:

1. early layers primarily encode pose, motion, and sensorimotor structure;
2. intermediate layers encode sign-level and compositional structure;
3. later layers develop more abstract semantic organization that aligns with representations from English, Chinese, Spanish, and multilingual language models.

## Candidate datasets

- **How2Sign (preferred):** continuous ASL video paired with English translations; suitable for testing emergence without mandatory gloss supervision.
- **PHOENIX-2014T:** continuous DGS with glosses and translations; useful as a benchmark, but gloss supervision can introduce symbolic or language leakage.
- **WLASL:** isolated ASL signs with word labels; useful for controlled lexical RSA, but less suitable for sentence- or language-level structure.

## Proposed representation pipeline

```text
sign video
  -> body, hand, and facial landmark extraction
  -> spatiotemporal pose/graph sequence
  -> graph, motion, or spatiotemporal Transformer encoder
  -> layer-wise latent representations
```

The preferred initial training regime is self-supervised learning on sign motion, such as masked or future representation prediction. Translation supervision can serve as a comparison condition, although it weakens claims that semantic organization emerged without explicit spoken-language supervision.

## Main analysis

Use representational similarity analysis (RSA) to compare layer-wise sign-model representational dissimilarity matrices with semantic RDMs derived from English, Chinese, Spanish, and multilingual language models.

Primary questions:

1. Do sign representations align with spoken-language semantic structure?
2. Is any alignment language-specific or shared across languages?
3. Does alignment increase across model depth while motion/pose alignment decreases?
4. Does self-supervised sign learning show semantic alignment without gloss or translation supervision?

## Longer-term extension

Compare model RDMs with neural RDMs from native signers to test whether artificial sign-language representations resemble the organization of human language and visuomotor networks.

## Important design risks

- English translations can make apparent cross-modal semantic alignment partly circular.
- Pose-only inputs may remove facial and mouth information that is linguistically meaningful.
- Sentence length, motion energy, signer identity, and video context can confound semantic RSA.
- Paired translations are not automatically a valid semantic stimulus set; evaluation items need controlled semantic contrasts or independent human similarity judgments.
- Claims of modality-independent concepts require comparisons against visual/motor and low-level nuisance RDMs, not only language-model RDMs.

## Working title

**From mBERT to Sign Language Models: Testing Modality-Independent Semantic Spaces with RSA**

## Selected first experiment

### Dataset: How2Sign B-F-H 2D keypoint sentence clips

Use the official frontal-view body-face-hands keypoint clips and the official
train/validation/test split. How2Sign contains more than 80 hours of continuous ASL and
provides sentence clips as well as manually re-aligned English translations. The English
text is deliberately excluded from self-supervised pretraining and should only be joined
later when constructing the semantic evaluation RDM.

The dataset is research-only and licensed CC BY-NC 4.0. Download it from the
[official How2Sign page](https://how2sign.github.io/index.html); raw data is intentionally
not tracked by this repository.

### Model: multi-stream masked-cluster Sign Transformer

The implementation follows the central design of
[SHuBERT (ACL 2025)](https://aclanthology.org/2025.acl-long.1397/): body, hands, and face
form separate input streams, continuous spans are masked, and a Transformer predicts
automatically clustered pose-and-motion targets for every stream. This is a compact,
from-scratch research implementation rather than a copy of the authors' model.

This model was selected because it:

- learns from continuous signing without gloss or translation labels;
- retains hands, face, and body as linguistically distinct streams;
- supports sentence context instead of isolated sign classification;
- exposes every Transformer layer for the proposed layer-wise RSA;
- is substantially cheaper to train on keypoints than an RGB video backbone.

## Repository layout

```text
configs/pretrain.json       main experiment configuration
src/sign_semantics/
  prepare.py                OpenPose JSON -> normalized sentence NPZ
  cluster.py                body/hands/face k-means pseudo-targets
  model.py                  multi-stream masked Transformer
  train.py                  pretraining and checkpointing
  extract.py                layer-wise sentence embeddings for RSA
tests/test_pipeline.py      synthetic end-to-end backward-pass test
```

## Installation

Python 3.10--3.13 is supported. A CUDA GPU is strongly recommended for the full model.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

### Google Colab

Use [notebooks/colab_pretrain.ipynb](notebooks/colab_pretrain.ipynb) for GPU training.
The notebook mounts Google Drive for archives and checkpoints, copies code and training
data to the Colab VM for faster small-file access, resumes from `last.pt`, and exports
layer-wise representations back to Drive. Before running it, either push this repository
to GitHub and set `REPO_URL`, or upload the project folder to
`MyDrive/sign_language_llm`.

## Data preparation

Download each official keypoint split into a separate directory. Each sentence clip must
ultimately be represented by one directory containing its frame-level OpenPose JSON files.
Then run:

```bash
sign-prepare \
  --input-root data/raw/how2sign/train \
  --output-root data/processed/clips/train \
  --manifest data/processed/train.jsonl

sign-prepare \
  --input-root data/raw/how2sign/val \
  --output-root data/processed/clips/val \
  --manifest data/processed/val.jsonl

sign-prepare \
  --input-root data/raw/how2sign/test \
  --output-root data/processed/clips/test \
  --manifest data/processed/test.jsonl
```

The converter selects the most confident detected signer, retains all 25 body, 42 hand,
and 70 facial landmarks, and normalizes coordinates per frame using the neck and shoulder
width. Missing detections remain confidence-weighted zeros.

## Create self-supervised targets

Fit the cluster vocabulary on the training split only:

```bash
sign-cluster \
  --manifest data/processed/train.jsonl \
  --output data/processed/cluster_centers.npz \
  --n-clusters 500
```

Every cluster feature includes normalized position, frame-to-frame velocity, and OpenPose
confidence. Validation and test frames must never be used to fit these centers.

## Pretrain

Review paths and batch size in `configs/pretrain.json`, then run:

```bash
sign-pretrain --config configs/pretrain.json
```

Resume an interrupted run with:

```bash
sign-pretrain \
  --config configs/pretrain.json \
  --resume outputs/pretrain/last.pt
```

The default model has 8 Transformer layers, hidden size 256, 8 attention heads, a maximum
window of 256 frames, and masks approximately 50% of valid frames in contiguous spans.
Checkpoints include the model, optimizer, scheduler, configuration, epoch, and best
validation loss.

## Export representations for RSA

```bash
sign-extract \
  --checkpoint outputs/pretrain/best.pt \
  --manifest data/processed/test.jsonl \
  --output outputs/representations/test_layers.npz
```

The output contains sentence IDs and one mean-pooled matrix per Transformer layer. Build
RDMs from these matrices and join them to translations by sentence ID outside the
self-supervised training pipeline.

## Test

```bash
python -m unittest discover -s tests -v
```

The test constructs synthetic OpenPose frames, prepares and loads a sentence, assigns
cluster targets, masks spans, performs a full forward/backward pass, and checks layer-wise
embedding extraction.

## Methodological constraints

- Do not use English translations or glosses during the primary pretraining condition.
- Fit preprocessing statistics and cluster centers using the training split only.
- Report signer-disjoint results if a defensible signer-disjoint split can be constructed.
- Include pose, motion energy, duration, signer identity, and text-length nuisance RDMs.
- Treat the translation-supervised model as a positive-control condition, not evidence of
  spontaneous cross-modal semantic emergence.
