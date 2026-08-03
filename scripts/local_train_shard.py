"""Download, train, and safely remove one YouTube-ASL shard on Apple MPS."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

SHARDS = {
    1: ("3dc57bf4-c5fb-491c-8ab2-a9215e0e2fe5", "75223dd5e7e9b6ccb9f34c5792fc1e6d"),
}
BASE = "https://lindat.mff.cuni.cz/repository/server/api/core/bitstreams"
TRAIN_ANNOTATION_ID = "f8460818-3605-4f05-9832-90ddc68f22e6"
DEV_ANNOTATION_ID = "d5d23d31-c93a-4752-8e5c-e20548f13da0"


def url(identifier: str) -> str:
    return f"{BASE}/{identifier}/content"


def download(target: Path, identifier: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    aria2 = shutil.which("aria2c")
    if aria2 is not None:
        subprocess.run(
            [
                aria2,
                "--continue=true",
                "--max-connection-per-server=8",
                "--split=8",
                "--min-split-size=16M",
                "--file-allocation=none",
                "--auto-file-renaming=false",
                "--allow-overwrite=false",
                "--max-tries=0",
                "--retry-wait=5",
                "--summary-interval=10",
                f"--dir={target.parent}",
                f"--out={target.name}",
                url(identifier),
            ],
            check=True,
        )
        return
    subprocess.run(
        [
            "curl",
            "--location",
            "--fail",
            "--retry",
            "5",
            "--retry-delay",
            "5",
            "--continue-at",
            "-",
            "--output",
            str(target),
            url(identifier),
        ],
        check=True,
    )


def md5(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=1, choices=SHARDS)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is not available in this process")

    project = Path(__file__).resolve().parents[1]
    raw_root = project / "data" / "raw"
    output_root = project / "outputs" / "local_mps" / f"shard_{args.shard:02d}"
    output_root.mkdir(parents=True, exist_ok=True)
    archive = raw_root / f"raw_keypoints_{args.shard}.zip"
    identifier, expected_md5 = SHARDS[args.shard]

    reusable = archive.stat().st_size if archive.exists() else 0
    available = shutil.disk_usage(project).free + reusable
    if available < 45 * 1024**3:
        raise RuntimeError(
            f"Need at least 45 GiB free or reusable; have {available / 1024**3:.1f} GiB"
        )

    train_annotations = raw_root / "YT.translations.train.json"
    dev_annotations = raw_root / "YT.translations.dev.json"
    for target, annotation_id in (
        (train_annotations, TRAIN_ANNOTATION_ID),
        (dev_annotations, DEV_ANNOTATION_ID),
    ):
        if not target.exists():
            print(f"Downloading {target.name}", flush=True)
            download(target, annotation_id)

    print(f"Downloading/resuming shard {args.shard}: {archive}", flush=True)
    download(archive, identifier)
    print("Verifying MD5", flush=True)
    actual_md5 = md5(archive)
    if actual_md5 != expected_md5:
        raise RuntimeError(f"MD5 mismatch: expected {expected_md5}, got {actual_md5}")

    config = json.loads((project / "configs" / "pretrain.json").read_text())
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
            "output_dir": str(output_root / "checkpoints"),
            "batch_size": args.batch_size,
            "epochs": 1,
            "amp": False,
            "progress_every": 1,
        }
    )
    config_path = output_root / "pretrain.json"
    config_path.write_text(json.dumps(config, indent=2))

    print("Starting one-epoch MPS training", flush=True)
    started = time.perf_counter()
    subprocess.run(
        [sys.executable, "-m", "sign_semantics.train", "--config", str(config_path)],
        cwd=project,
        check=True,
    )
    checkpoint = output_root / "checkpoints" / "last.pt"
    if not checkpoint.exists() or checkpoint.stat().st_size == 0:
        raise RuntimeError("Training returned successfully but last.pt is missing or empty")

    manifest = {
        "shard": args.shard,
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "training_seconds": round(time.perf_counter() - started, 1),
        "source_md5": actual_md5,
    }
    (output_root / "completed.json").write_text(json.dumps(manifest, indent=2))
    archive.unlink()
    print(
        f"Completed shard {args.shard}; checkpoint={checkpoint}; "
        f"deleted_archive={archive}",
        flush=True,
    )


if __name__ == "__main__":
    main()
