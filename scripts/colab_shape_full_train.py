"""One-click resumable full training for the shape-aware contrastive model."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from google.colab import files


SHARDS = {
    1: ("3dc57bf4-c5fb-491c-8ab2-a9215e0e2fe5", "75223dd5e7e9b6ccb9f34c5792fc1e6d"),
    2: ("bfa38e27-bd46-48ae-bf9a-eb1eafaabb95", "c891a51901ca17fa6f42529e5657df67"),
    3: ("0c59bda5-908d-4194-93bf-13e13de2ef10", "0713086c142d52c31cff1b4be9f4f82a"),
    4: ("1a4ace36-ed9a-4bfb-a80f-483c0463e02d", "890dd2fa779b56cd90c6ae39fa67faef"),
    5: ("6b405702-9f74-4729-8202-55eca36adaea", "e935642e8bb5f9b594af74a8ba75d97f"),
    6: ("3b4ef094-bc95-4efb-b8e0-3bc4f63e57b0", "403958646966402245f69f9d473c4346"),
    7: ("74e99da7-c580-4fdf-8363-8024d7a7adf1", "786fe0067d1e4d8665b1ddbb8628a17c"),
    8: ("43a9146e-0abf-47ec-a3b3-90c01a4d9380", "b6bcc2e8517c2dcf8347347bbd74800c"),
    9: ("05385788-d459-4f35-92b4-4908b9d86de6", "9d52caab1aa2db4218188819485f92ab"),
    10: ("d96b874e-72fa-4830-b2f0-a072bb6be31d", "0019b603f9ebd7594fce8fae8dc65167"),
}
BASE = "https://lindat.mff.cuni.cz/repository/server/api/core/bitstreams"
TRAIN_ANNOTATION_ID = "f8460818-3605-4f05-9832-90ddc68f22e6"
DEV_ANNOTATION_ID = "d5d23d31-c93a-4752-8e5c-e20548f13da0"
PROJECT = Path("/content/youtube-asl-skeleton-bert")
LOCAL_ROOT = Path("/content/youtube_asl_data")
RUN_ROOT = Path("/content/sign_semantics_youtube_asl/full_shape_contrastive")
CHECKPOINT_ROOT = RUN_ROOT / "checkpoints"


def url(identifier: str) -> str:
    return f"{BASE}/{identifier}/content"


def print_disk(label: str) -> None:
    usage = shutil.disk_usage("/content")
    print(
        f"{label}: used={usage.used / 1024**3:.1f} GiB, "
        f"free={usage.free / 1024**3:.1f} GiB",
        flush=True,
    )


def md5(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fast_download_command(source: str, target: Path) -> list[str]:
    target.parent.mkdir(parents=True, exist_ok=True)
    aria2 = shutil.which("aria2c")
    if aria2:
        return [
            aria2,
            "--continue=true",
            "--max-connection-per-server=16",
            "--split=16",
            "--min-split-size=16M",
            "--file-allocation=none",
            "--auto-file-renaming=false",
            "--allow-overwrite=false",
            "--max-tries=0",
            "--retry-wait=5",
            "--summary-interval=10",
            f"--dir={target.parent}",
            f"--out={target.name}",
            source,
        ]
    return ["wget", "-c", "--show-progress", "-O", str(target), source]


def fast_download(source: str, target: Path) -> None:
    command = fast_download_command(source, target)
    print(f"Downloading with: {Path(command[0]).name} {target.name}", flush=True)
    subprocess.run(command, check=True)


def restore_uploaded_bundle() -> None:
    candidates = sorted(
        Path("/content").glob("youtube_asl_shape_contrastive_resume_after_shard_*.zip")
    )
    canonical = Path("/content/youtube_asl_shape_contrastive_full_resume.zip")
    if canonical.exists():
        candidates.append(canonical)
    if not candidates or (RUN_ROOT / "completed_shards.json").exists():
        return
    bundle = max(candidates, key=lambda path: path.stat().st_mtime)
    with zipfile.ZipFile(bundle) as handle:
        handle.extractall(RUN_ROOT)
    print(f"Restored checkpoint state from uploaded bundle: {bundle}", flush=True)


def download_resume_bundle(state_path: Path, shard: int) -> None:
    bundle = Path(
        f"/content/youtube_asl_shape_contrastive_resume_after_shard_{shard:02d}.zip"
    )
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.write(state_path, "completed_shards.json")
        # last.pt is the complete epoch/shard checkpoint and is sufficient to resume.
        for name in ("last.pt", "best.pt", "metrics.jsonl", "metrics.csv"):
            source = CHECKPOINT_ROOT / name
            if source.exists():
                handle.write(source, f"checkpoints/{name}")
    if not (CHECKPOINT_ROOT / "last.pt").exists():
        raise RuntimeError("Refusing to download a bundle without last.pt")
    print(f"Downloading resumable epoch bundle: {bundle}", flush=True)
    files.download(str(bundle))
    for older in Path("/content").glob(
        "youtube_asl_shape_contrastive_resume_after_shard_*.zip"
    ):
        if older != bundle:
            older.unlink()


def write_config(
    archive: Path,
    train_annotations: Path,
    dev_annotations: Path,
    epochs: int,
) -> Path:
    source = PROJECT / "configs/pretrain_shape_contrastive.json"
    if not source.exists():
        raise FileNotFoundError(
            f"Missing {source}; make sure the notebook cloned the shape-aware branch"
        )
    config = json.loads(source.read_text())
    config["data"].update(
        {
            "train_archive": str(archive),
            "val_archive": str(archive),
            "train_annotations": str(train_annotations),
            "val_annotations": str(dev_annotations),
            "max_frames": 256,
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
    path = Path("/content/colab_shape_contrastive.json")
    path.write_text(json.dumps(config, indent=2))
    return path


def train_once(config_path: Path, resume: Path | None) -> None:
    command = [
        sys.executable,
        "-u",
        "-m",
        "sign_semantics.shape_train",
        "--config",
        str(config_path),
    ]
    if resume is not None:
        command += ["--resume", str(resume)]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    if not PROJECT.exists():
        raise FileNotFoundError(f"Project was not cloned: {PROJECT}")
    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    restore_uploaded_bundle()
    state_path = RUN_ROOT / "completed_shards.json"
    completed = json.loads(state_path.read_text()) if state_path.exists() else []
    print(
        "Experiment: part-specific ST-GCN + 6-layer temporal Transformer; "
        "contrastive sentence training; no text, gloss, boundaries, or latent units.",
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
        print(
            f"Full progress: {len(completed)}/{len(SHARDS)} shards; "
            f"starting shard {shard}.",
            flush=True,
        )
        archive = LOCAL_ROOT / f"raw_keypoints_{shard}.zip"
        reusable = archive.stat().st_size if archive.exists() else 0
        available = shutil.disk_usage("/content").free + reusable
        if available < 45 * 1024**3:
            raise RuntimeError(
                f"Need 45 GiB free or reusable; have {available / 1024**3:.1f} GiB"
            )
        print_disk(f"Before shard {shard} download")
        if shard in prefetch_jobs:
            process, log_path = prefetch_jobs.pop(shard)
            print(f"Waiting for prefetched shard {shard} if needed...", flush=True)
            if process.wait() != 0:
                print(f"Prefetch failed; resuming foreground download: {log_path}")
                fast_download(url(identifier), archive)
        else:
            fast_download(url(identifier), archive)
        if md5(archive) != expected_md5:
            raise RuntimeError(f"MD5 mismatch for shard {shard}")
        print_disk(f"After shard {shard} download")

        future = [number for number in SHARDS if number > shard and number not in completed]
        if future:
            next_shard = future[0]
            next_archive = LOCAL_ROOT / f"raw_keypoints_{next_shard}.zip"
            next_reusable = next_archive.stat().st_size if next_archive.exists() else 0
            prefetch_available = shutil.disk_usage("/content").free + next_reusable
            if prefetch_available >= 45 * 1024**3:
                log_path = Path(f"/content/prefetch_shape_shard_{next_shard:02d}.log")
                log_handle = log_path.open("w")
                process = subprocess.Popen(
                    fast_download_command(url(SHARDS[next_shard][0]), next_archive),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
                log_handle.close()
                prefetch_jobs[next_shard] = (process, log_path)
                print(
                    f"Background prefetch started for shard {next_shard}: {log_path}",
                    flush=True,
                )
            else:
                print("Skipping prefetch to preserve 45 GiB free.", flush=True)

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
        print_disk(f"After deleting shard {shard} data")
        elapsed = time.perf_counter() - started
        remaining = len(SHARDS) - len(completed)
        eta_hours = elapsed / max(completed_this_session, 1) * remaining / 3600
        print(
            f"Completed shard {shard}; full progress: {len(completed)}/10; "
            f"session ETA: {eta_hours:.1f} hours.",
            flush=True,
        )

    print("All ten shape-aware training epochs are complete.", flush=True)


if __name__ == "__main__":
    main()
