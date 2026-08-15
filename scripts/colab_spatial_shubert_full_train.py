"""One-click, shard-wise training for skeleton-only Spatial-GCN SHuBERT."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from google.colab import files

from colab_shape_full_train import (
    DEV_ANNOTATION_ID,
    SHARDS,
    TRAIN_ANNOTATION_ID,
    fast_download,
    fast_download_command,
    md5,
    print_disk,
    url,
)


PROJECT = Path("/content/youtube-asl-skeleton-bert")
LOCAL_ROOT = Path("/content/youtube_asl_data")
RUN_ROOT = Path("/content/sign_semantics_youtube_asl/full_spatial_shubert")
CHECKPOINT_ROOT = RUN_ROOT / "checkpoints"
CODEBOOK_PATH = RUN_ROOT / "spatial_shubert_codebooks.npz"


def restore_uploaded_bundle() -> None:
    candidates = sorted(Path("/content").glob("spatial_shubert_resume_after_shard_*.zip"))
    if not candidates or (RUN_ROOT / "completed_shards.json").exists():
        return
    bundle = max(candidates, key=lambda path: path.stat().st_mtime)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle) as handle:
        handle.extractall(RUN_ROOT)
    print(f"Restored Spatial-SHuBERT state from {bundle}", flush=True)


def download_resume_bundle(state_path: Path, shard: int) -> None:
    bundle = Path(f"/content/spatial_shubert_resume_after_shard_{shard:02d}.zip")
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.write(state_path, "completed_shards.json")
        handle.write(CODEBOOK_PATH, CODEBOOK_PATH.name)
        for name in ("last.pt", "best.pt", "metrics.jsonl", "metrics.csv"):
            source = CHECKPOINT_ROOT / name
            if source.exists():
                handle.write(source, f"checkpoints/{name}")
    if not (CHECKPOINT_ROOT / "last.pt").exists():
        raise RuntimeError("Refusing to download a resume bundle without last.pt")
    print(f"Downloading resumable epoch bundle: {bundle}", flush=True)
    files.download(str(bundle))
    for older in Path("/content").glob("spatial_shubert_resume_after_shard_*.zip"):
        if older != bundle:
            older.unlink()


def fit_codebooks(archive: Path, annotations: Path) -> None:
    if CODEBOOK_PATH.exists():
        print(f"Reusing frozen codebooks: {CODEBOOK_PATH}", flush=True)
        return
    print(
        "Fitting four frame-local skeleton codebooks on shard 1. "
        "This happens once; no glosses or translations are read.",
        flush=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "sign_semantics.skeleton_codebooks",
            "--archive",
            str(archive),
            "--annotations",
            str(annotations),
            "--output",
            str(CODEBOOK_PATH),
            "--max-sampled-frames",
            "100000",
            "--clusters",
            "100",
            "--batch-size",
            "32",
            "--num-workers",
            "2",
        ],
        check=True,
    )


def write_config(
    archive: Path, train_annotations: Path, dev_annotations: Path, epochs: int
) -> Path:
    source = PROJECT / "configs/pretrain_spatial_shubert.json"
    if not source.exists():
        raise FileNotFoundError(f"Missing {source}; rerun the clone/install cells")
    config = json.loads(source.read_text())
    config["objective"]["codebook_path"] = str(CODEBOOK_PATH)
    config["data"].update(
        {
            "train_archive": str(archive),
            "val_archive": str(archive),
            "train_annotations": str(train_annotations),
            "val_annotations": str(dev_annotations),
            "num_workers": 2,
            "prefetch_factor": 2,
        }
    )
    config["training"].update(
        {
            "output_dir": str(CHECKPOINT_ROOT),
            "batch_size": 16,
            "epochs": epochs,
            "amp": True,
            "progress_every": 1,
            "save_every": 0,
        }
    )
    path = Path("/content/colab_spatial_shubert.json")
    path.write_text(json.dumps(config, indent=2))
    return path


def train_once(config_path: Path, resume: Path | None) -> None:
    command = [
        sys.executable,
        "-u",
        "-m",
        "sign_semantics.spatial_shubert_train",
        "--config",
        str(config_path),
    ]
    if resume is not None:
        command += ["--resume", str(resume)]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    restore_uploaded_bundle()
    state_path = RUN_ROOT / "completed_shards.json"
    completed = json.loads(state_path.read_text()) if state_path.exists() else []
    print(
        "Experiment: skeleton-only multi-stream Spatial GCN + 6-layer Transformer; "
        "SHuBERT-style masked cluster prediction; no RGB, text, gloss, or boundaries.",
        flush=True,
    )

    train_annotations = LOCAL_ROOT / "YT.translations.train.json"
    dev_annotations = LOCAL_ROOT / "YT.translations.dev.json"
    for target, identifier in (
        (train_annotations, TRAIN_ANNOTATION_ID),
        (dev_annotations, DEV_ANNOTATION_ID),
    ):
        if not target.exists():
            fast_download(url(identifier), target)

    started = time.perf_counter()
    completed_this_session = 0
    prefetch_jobs: dict[int, tuple[subprocess.Popen, Path]] = {}
    for shard, (identifier, expected_md5) in SHARDS.items():
        if shard in completed:
            print(f"Shard {shard} already completed; skipping.", flush=True)
            continue
        print(f"Full progress: {len(completed)}/10 shards; starting shard {shard}.", flush=True)
        archive = LOCAL_ROOT / f"raw_keypoints_{shard}.zip"
        print_disk(f"Before shard {shard} download")
        if shard in prefetch_jobs:
            process, log_path = prefetch_jobs.pop(shard)
            if process.wait() != 0:
                print(f"Prefetch failed ({log_path}); resuming in foreground.", flush=True)
                fast_download(url(identifier), archive)
        else:
            fast_download(url(identifier), archive)
        if md5(archive) != expected_md5:
            raise RuntimeError(f"MD5 mismatch for shard {shard}")
        print_disk(f"After shard {shard} download")

        if shard == 1:
            fit_codebooks(archive, train_annotations)

        future = [number for number in SHARDS if number > shard and number not in completed]
        if future and shutil.disk_usage("/content").free >= 45 * 1024**3:
            next_shard = future[0]
            next_archive = LOCAL_ROOT / f"raw_keypoints_{next_shard}.zip"
            log_path = Path(f"/content/prefetch_spatial_shubert_{next_shard:02d}.log")
            log_handle = log_path.open("w")
            process = subprocess.Popen(
                fast_download_command(url(SHARDS[next_shard][0]), next_archive),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            log_handle.close()
            prefetch_jobs[next_shard] = (process, log_path)
            print(f"Background prefetch started for shard {next_shard}: {log_path}", flush=True)

        config_path = write_config(
            archive, train_annotations, dev_annotations, len(completed) + 1
        )
        resume = CHECKPOINT_ROOT / "last.pt"
        train_once(config_path, resume if completed and resume.exists() else None)
        completed.append(shard)
        completed_this_session += 1
        state_path.write_text(json.dumps(completed))
        download_resume_bundle(state_path, shard)
        archive.unlink()
        elapsed = time.perf_counter() - started
        remaining = len(SHARDS) - len(completed)
        eta_hours = elapsed / max(completed_this_session, 1) * remaining / 3600
        print(
            f"Completed shard {shard}; local shard removed; progress={len(completed)}/10; "
            f"session ETA={eta_hours:.1f} hours.",
            flush=True,
        )
    print("All ten Spatial-SHuBERT epochs are complete.", flush=True)


if __name__ == "__main__":
    main()
