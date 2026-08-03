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

The data is not committed to GitHub and the Colab workflow never puts raw data
in Google Drive. A shard stays temporarily under `/content`, is read directly
from the ZIP without extraction, and is deleted only after its checkpoint,
metrics, and completion state are verified in Drive (or in the fallback resume
bundle). Roughly 165 GB of free Colab disk is enough for this one-shard-at-a-time
workflow, but not for all ten roughly 35 GB shards at once.

## Colab training

Open [`notebooks/colab_pretrain.ipynb`](notebooks/colab_pretrain.ipynb), select a
GPU runtime, and run the cells. The notebook:

1. verifies the GPU, requires distinct Colab and Drive account emails, mounts
   the selected Drive account, and checks a persistent account marker;
2. clones this GitHub repository into `/content`;
3. installs aria2 and downloads the small official train/dev annotations;
4. downloads with 16-connection resumable aria2 and, while shard N trains,
   prefetches shard N+1 in the background;
5. trains directly from the current ZIP while keeping at most two shards on
   local disk;
6. saves `last.pt`, `best.pt`, `metrics.jsonl`, and `metrics.csv` locally, then
   copies and size-verifies them in Drive after every shard;
7. updates `completed_shards.json` in Drive and only then removes the local
   shard and prints disk usage;
8. falls back to a browser-downloaded resume bundle if Drive mounting fails.

During training, a live progress bar and flushed `training_progress` JSON records
report processed clips, percentage, throughput, loss, optimizer steps, and ETA.
At the end of every epoch, the trainer prints a readable summary and one
machine-readable `epoch_metrics` JSON record. Both metric files contain
train/validation total, body, hands, and face losses; MPJPE; RMSE; PCK@0.1 and
PCK@0.2; elapsed time; throughput; and peak GPU memory. The loader uses fast
JSON decoding, reads only the selected 37 facial landmarks, and prefetches
batches with persistent workers to keep the GPU supplied.

At startup the notebook restores the latest completion state, checkpoints, and
metric history from Drive. The Drive account can differ from the account that
supplies the Colab runtime: the notebook rejects equal email entries and checks
a marker under `MyDrive/sign_semantics_youtube_asl/`. Google still requires the
user to confirm the authorisation dialog because `drive.mount()` does not expose
the mounted email to Python. If Google forces the Colab account, the target
Drive account should share an editor folder with the Colab account, which adds
that folder to My Drive as a shortcut named `sign_semantics_youtube_asl`.
Checkpoint writes then follow the shortcut into the shared folder. If Drive is
unavailable, save the newest fallback resume ZIP and upload it later as
`/content/youtube_asl_full_resume.zip`. Dataset shards remain temporary and are
never copied to Drive.

No verified public mirror of the keypoint release is currently available. The
prefetch pipeline reduces end-to-end wall time by overlapping the next roughly
35 GB download with current-shard training; it does not increase the official
server's raw bandwidth. With roughly 165 GB free, two simultaneous shards plus
checkpoints fit safely while preserving a 45 GB free-space guard.

The committed Colab notebook now defaults to `MODE = "full"` after the pilot was
successfully completed. For a fresh environment, set `MODE = "pilot"` to
verify GPU, data, and loss before returning to full mode. If Colab disconnects
between shards, rerunning the notebook restores Drive state (or the uploaded
fallback resume bundle) and continues with the next shard.

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

On an Apple-silicon Mac, shard 1 can be downloaded, trained with MPS, and
deleted only after a non-empty checkpoint is verified with:

```bash
python scripts/local_train_shard.py --shard 1
```

Interrupted downloads resume. A failed download, checksum, or training run
keeps the shard ZIP so that it is not downloaded again.

## Later word-level RSA

Word-level RSA remains a held-out analysis. A separate isolated-word or manually
time-aligned evaluation set should be converted with the same 104-point layout.
Those word labels/boundaries must never enter sentence-level pretraining. The
resulting Skeleton-BERT word RDM can then be compared with layer-wise mBERT RDMs,
alongside raw-kinematic, duration, signer, and untrained-model controls.
