# Skeleton BERT for Sign Semantics

This repository tests whether word-level semantic geometry appears in a model trained only
to reconstruct masked skeleton trajectories from continuous sign-language sentences.

The skeleton model is a small, randomly initialized Hugging Face `BertModel`. It receives
no text, gloss, word boundary, translation, class label, or pretrained language weight
during training. Gloss and time boundaries are used only after training to locate word
tokens in held-out sentences for RSA.

## Pipeline

```text
TRAINING
How2Sign sentence-level OpenPose JSON
  -> body + hands + face arrays
  -> per-frame neck/shoulder normalization
  -> random sentence window and padding
  -> mask contiguous spans of skeleton frames
  -> Linear(411 -> 256)
  -> randomly initialized 6-layer BERT
  -> reconstruct masked coordinates/confidence and within-span velocity

WORD-LEVEL EVALUATION
held-out full sentence
  -> unmasked Skeleton BERT hidden states at every layer
  -> test-only gloss timestamps select frames belonging to each word token
  -> average frames within token
  -> average repeated tokens into one vector per word type
  -> one sign RDM per Skeleton BERT layer

TEXT REFERENCE
same concept IDs and multilingual text forms
  -> pretrained mBERT
  -> one text RDM per mBERT layer
  -> Spearman RSA + word-label permutation test
```

## Why this is self-supervised

The input and target are both derived from the same skeleton sentence. The target for a
masked span is its original normalized trajectory. The training process never loads the
word-boundary file or the English translations. The result can therefore test semantic
organization without lexical or semantic training labels, subject to appropriate
kinematic controls.

## Model

The default model uses:

- one flattened frame stream containing 25 body, 42 hand, and 70 face landmarks;
- `(x, y, confidence)` for each landmark, or 411 input values per frame;
- a learned linear skeleton projection;
- a single learned mask embedding;
- a 6-layer BERT initialized from scratch;
- hidden size 256, 8 attention heads, and intermediate size 1024;
- separate reconstruction slices only for balanced body/hands/face losses.

The model calls `BertModel(inputs_embeds=...)`; it never tokenizes skeleton data and never
loads mBERT weights. mBERT is a separate analysis-only reference model.

## Repository layout

```text
configs/pretrain.json           experiment configuration
notebooks/colab_pretrain.ipynb  Colab GPU workflow
src/sign_semantics/
  prepare.py                    OpenPose JSON -> normalized sentence NPZ
  data.py                       variable-length windowing, padding, frame mapping
  masking.py                    contiguous temporal span masking
  model.py                      randomly initialized Skeleton BERT
  train.py                      masked trajectory pretraining
  extract.py                    optional sentence-level representations
  word_extract.py               test-only boundary pooling into word types
  text_embed.py                 layer-wise mBERT concept representations
  rsa.py                        RDMs, layer-wise RSA, permutation tests
tests/test_pipeline.py          synthetic training and RSA tests
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

For Colab, use [notebooks/colab_pretrain.ipynb](notebooks/colab_pretrain.ipynb). Keep
archives and checkpoints in Google Drive, but extract the many small JSON/NPZ files into
the Colab VM under `/content`.

## 1. Prepare skeleton sentences

Download the official frontal body-face-hands keypoint splits from the
[How2Sign dataset page](https://how2sign.github.io/). The dataset is CC BY-NC 4.0 and is
not redistributed by this repository.

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

Each output NPZ contains `body`, `hands`, and `face` arrays. The manifest contains only a
clip ID, file path, and number of frames. It contains no text or gloss.

## 2. Pretrain Skeleton BERT

```bash
sign-pretrain --config configs/pretrain.json
```

Resume after interruption:

```bash
sign-pretrain \
  --config configs/pretrain.json \
  --resume outputs/pretrain/last.pt
```

The default objective masks about 40% of valid frames in spans averaging 20 frames. The
loss averages body, hands, and face reconstruction so the 70 face landmarks do not
overwhelm the smaller streams. A velocity term is applied to adjacent masked frames.

## 3. Extract contextual word representations

Create a test-only JSONL boundary file. Frame indices refer to the original sentence clip;
`end_frame` is exclusive.

```json
{"clip_id":"clip_001","gloss":"DOG","start_frame":42,"end_frame":68}
{"clip_id":"clip_002","gloss":"DOG","start_frame":15,"end_frame":39}
{"clip_id":"clip_003","gloss":"CAT","start_frame":70,"end_frame":91}
```

The boundary file is never referenced by the training command.

```bash
sign-extract-words \
  --checkpoint outputs/pretrain/best.pt \
  --manifest data/processed/test.jsonl \
  --boundaries data/annotations/test_word_boundaries.jsonl \
  --min-tokens 5 \
  --output outputs/representations/sign_words.npz
```

For each BERT layer, the command encodes the complete sentence without masking, averages
hidden frames within each word boundary, and averages repeated occurrences into one vector
per gloss. Long sentences retain a mapping from sampled model frames to original frames.

## 4. Create mBERT word references

Create a CSV with the same concept IDs. `text` may contain a bare word or a standardized
short context. Use the same contextual template across concepts.

```csv
id,text
DOG,dog
CAT,cat
HOUSE,house
```

```bash
sign-embed-text \
  --concepts data/annotations/concepts_en.csv \
  --model bert-base-multilingual-cased \
  --output outputs/representations/mbert_en_words.npz
```

Repeat with Chinese, Spanish, or other translations while keeping the `id` column fixed.
This command is analysis-only and downloads the selected text model from Hugging Face.

## 5. Run layer-wise word RSA

```bash
sign-rsa \
  --sign outputs/representations/sign_words.npz \
  --text outputs/representations/mbert_en_words.npz \
  --metric correlation \
  --permutations 1000 \
  --output outputs/rsa/sign_vs_mbert_en.csv
```

The tool aligns the two files by concept ID, constructs one RDM per layer, correlates RDM
upper triangles with Spearman's rho, and estimates a two-sided word-label permutation
p-value. The CSV contains every Skeleton-BERT-layer by mBERT-layer comparison.

## Required scientific controls

The implemented RSA tests alignment but does not by itself establish semantic causality.
A confirmatory analysis should additionally compare or partial out:

- raw normalized skeleton distance;
- hand trajectory and velocity distance;
- word duration;
- signer identity;
- an untrained Skeleton BERT;
- independently split token averages where enough signers are available.

Evidence for semantic emergence requires mBERT/human-semantic alignment that survives
these kinematic and sampling controls.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests create synthetic OpenPose sentences, run masked Skeleton-BERT reconstruction and
backpropagation, validate the original-frame mapping and word-boundary schema, and execute
a small layer-wise RSA with permutation testing.
