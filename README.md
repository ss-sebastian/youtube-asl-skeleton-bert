# YouTube-ASL Skeleton BERT

Self-supervised learning from continuous American Sign Language skeleton
trajectories. The model is a small BERT-style Transformer initialized from
scratch. It sees sentence-level skeleton motion only: no glosses, translations,
word boundaries, text embeddings, or pretrained language weights are used as
training targets.

## Current training flow

```text
YouTube-ASL sentence keypoint JSON
  -> 25 upper-body pose + 42 hand + 37 face points
  -> shoulder-centred, shoulder-scaled 2D trajectory
  -> 104 points x (x, y) = 208 values per frame
  -> mask contiguous temporal spans
  -> randomly initialized 6-layer Skeleton BERT
  -> reconstruct masked coordinates and velocity
```

Missing MediaPipe landmarks are represented by a separate observation mask and
are excluded from reconstruction loss. Sentence translations are present in the
official annotation files, but the loader reads only `clip_order` identifiers to
construct the official train/dev split.

## Dataset

The project uses the official
[YouTube-ASL Clip Keypoint Dataset](http://hdl.handle.net/11234/1-5898):

- 390,547 sentence-level ASL keypoint clips;
- ten ZIP shards of roughly 37 GB each (about 374 GB compressed in total);
- frame-wise MediaPipe pose, hands, and face landmarks;
- CC BY 4.0.

The data is not committed to GitHub and the Colab workflow does not put it in
Google Drive. A shard stays temporarily under `/content`, is read directly from
the ZIP without extraction, and is deleted after that stage finishes.

## Colab training

Open [`notebooks/colab_pretrain.ipynb`](notebooks/colab_pretrain.ipynb), select a
GPU runtime, and run the cells. The notebook:

1. mounts Google Drive for model outputs only;
2. clones this GitHub repository into `/content`;
3. downloads the small official train/dev annotation files into `/content`;
4. downloads one keypoint shard at a time into `/content`;
5. trains directly from that ZIP;
6. saves `last.pt`, `best.pt`, logs, and completion state under
   `MyDrive/sign_semantics_youtube_asl/`;
7. removes the local shard after successful training.

The committed Colab notebook now defaults to `MODE = "full"` after the pilot was
successfully completed. For a fresh environment, set `MODE = "pilot"` to
verify GPU, data, loss, and Drive output before returning to full mode. If Colab
disconnects between shards, rerunning the notebook reads the completion state
from Drive and continues with the next shard.

Ten 37 GB downloads plus training are not expected to finish in one free Colab
session. The notebook is intentionally resumable across sessions.

## Local installation and tests

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v
```

## Manual training command

Edit `configs/pretrain.json` so that its archive and annotation paths point to
one local shard, then run:

```bash
sign-pretrain --config configs/pretrain.json
```

To continue from a prior stage:

```bash
sign-pretrain --config configs/pretrain.json --resume /path/to/last.pt
```

## Later word-level RSA

Word-level RSA remains a held-out analysis. A separate isolated-word or manually
time-aligned evaluation set should be converted with the same 104-point layout.
Those word labels/boundaries must never enter sentence-level pretraining. The
resulting Skeleton-BERT word RDM can then be compared with layer-wise mBERT RDMs,
alongside raw-kinematic, duration, signer, and untrained-model controls.
