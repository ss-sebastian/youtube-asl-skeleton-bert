"""Run resumable YouTube-ASL full training in a Colab background process."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

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


def url(identifier: str) -> str:
    return f"{BASE}/{identifier}/content"


def md5(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(target: Path, identifier: str) -> None:
    aria2 = shutil.which("aria2c")
    if aria2 is not None:
        command = [
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
            url(identifier),
        ]
    else:
        command = [
            "wget",
            "-c",
            "--show-progress",
            "-O",
            str(target),
            url(identifier),
        ]
    subprocess.run(command, check=True)


def main() -> None:
    project = Path("/content/youtube-asl-skeleton-bert")
    local_root = Path("/content/youtube_asl_data")
    local_root.mkdir(parents=True, exist_ok=True)
    drive_root = Path("/content/drive/MyDrive")
    persist_root = (
        drive_root / "sign_semantics_youtube_asl"
        if drive_root.exists()
        else Path("/content/sign_semantics_youtube_asl")
    )
    run_root = persist_root / "full"
    checkpoint_root = run_root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    state_path = run_root / "completed_shards.json"
    completed = json.loads(state_path.read_text()) if state_path.exists() else []

    train_annotations = local_root / "YT.translations.train.json"
    dev_annotations = local_root / "YT.translations.dev.json"
    for target, identifier in (
        (train_annotations, TRAIN_ANNOTATION_ID),
        (dev_annotations, DEV_ANNOTATION_ID),
    ):
        if not target.exists():
            download(target, identifier)

    started = time.perf_counter()
    completed_this_session = 0
    for shard, (identifier, expected_md5) in SHARDS.items():
        if shard in completed:
            print(f"Shard {shard} already completed; skipping.", flush=True)
            continue
        print(
            f"Full progress: {len(completed)}/{len(SHARDS)} shards; "
            f"starting shard {shard}.",
            flush=True,
        )
        archive = local_root / f"raw_keypoints_{shard}.zip"
        reusable = archive.stat().st_size if archive.exists() else 0
        available = shutil.disk_usage("/content").free + reusable
        if available < 45 * 1024**3:
            raise RuntimeError(f"Need 45 GB free or reusable; have {available / 1024**3:.1f} GB")
        download(archive, identifier)
        actual_md5 = md5(archive)
        if actual_md5 != expected_md5:
            raise RuntimeError(f"MD5 mismatch for shard {shard}: {actual_md5}")

        config = json.loads((project / "configs/pretrain.json").read_text())
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
                "output_dir": str(checkpoint_root),
                "batch_size": 16,
                "progress_every": 1,
                "epochs": len(completed) + 1,
                "amp": True,
                "save_every": 0,
            }
        )
        config_path = Path("/content/colab_pretrain.json")
        config_path.write_text(json.dumps(config, indent=2))
        command = [
            sys.executable,
            "-u",
            "-m",
            "sign_semantics.train",
            "--config",
            str(config_path),
        ]
        resume = checkpoint_root / "last.pt"
        if completed and resume.exists():
            command += ["--resume", str(resume)]
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, check=True)

        completed.append(shard)
        completed_this_session += 1
        state_path.write_text(json.dumps(completed))
        bundle = Path(f"/content/youtube_asl_full_resume_after_shard_{shard:02d}.zip")
        with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as handle:
            handle.write(state_path, "completed_shards.json")
            for name in ("last.pt", "best.pt"):
                checkpoint = checkpoint_root / name
                if checkpoint.exists():
                    handle.write(checkpoint, f"checkpoints/{name}")
            metrics = checkpoint_root / "metrics.jsonl"
            if metrics.exists():
                handle.write(metrics, "checkpoints/metrics.jsonl")
            metrics_csv = checkpoint_root / "metrics.csv"
            if metrics_csv.exists():
                handle.write(metrics_csv, "checkpoints/metrics.csv")
        archive.unlink()
        elapsed = time.perf_counter() - started
        remaining = len(SHARDS) - len(completed)
        eta_hours = elapsed / max(completed_this_session, 1) * remaining / 3600
        print(
            f"Completed shard {shard}; checkpoint={resume}; resume_bundle={bundle}; "
            f"full_progress={len(completed)}/{len(SHARDS)}; "
            f"session_eta_hours={eta_hours:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
