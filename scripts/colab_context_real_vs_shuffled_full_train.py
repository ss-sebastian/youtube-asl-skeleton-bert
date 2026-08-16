"""One-click paired real/shuffled context Spatial-SHuBERT training in Colab."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

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
RUN_ROOT = Path("/content/sign_semantics_youtube_asl/context_real_vs_shuffled")
CODEBOOK_PATH = RUN_ROOT / "spatial_shubert_codebooks.npz"
DIAGNOSTIC_ROOT = RUN_ROOT / "context_manipulation"
CONDITIONS = ("real", "shuffled")


def checkpoint_root(condition: str) -> Path:
    return RUN_ROOT / f"spatial_shubert_{condition}_context" / "checkpoints"


def restore_uploaded_bundle() -> None:
    candidates = sorted(Path("/content").glob("context_real_vs_shuffled_after_shard_*.zip"))
    if not candidates or (RUN_ROOT / "completed_shards.json").exists():
        return
    bundle = max(candidates, key=lambda path: path.stat().st_mtime)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle) as handle:
        handle.extractall(RUN_ROOT)
    print(f"Restored paired context experiment from {bundle}", flush=True)


def make_bundle(state_path: Path, shard: int) -> None:
    bundle = Path(f"/content/context_real_vs_shuffled_after_shard_{shard:02d}.zip")
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.write(state_path, "completed_shards.json")
        handle.write(CODEBOOK_PATH, CODEBOOK_PATH.name)
        smoke_state = RUN_ROOT / "smoke_complete.json"
        if smoke_state.exists():
            handle.write(smoke_state, smoke_state.name)
        for condition in CONDITIONS:
            root = checkpoint_root(condition)
            for name in ("last.pt", "best.pt", "metrics.jsonl", "metrics.csv"):
                source = root / name
                if source.exists():
                    handle.write(source, f"spatial_shubert_{condition}_context/checkpoints/{name}")
            for mapping in sorted(root.glob("context_mapping_epoch_*.jsonl")):
                handle.write(
                    mapping,
                    f"spatial_shubert_{condition}_context/checkpoints/{mapping.name}",
                )
        if DIAGNOSTIC_ROOT.exists():
            for source in DIAGNOSTIC_ROOT.rglob("*"):
                if source.is_file():
                    handle.write(source, source.relative_to(RUN_ROOT))
    for condition in CONDITIONS:
        if not (checkpoint_root(condition) / "last.pt").exists():
            raise RuntimeError(f"Refusing to bundle without {condition} last.pt")
    print(f"Checkpoint bundle ready: {bundle}", flush=True)
    print(f"COLAB_DOWNLOAD={bundle}", flush=True)


def fit_codebooks(archive: Path, annotations: Path) -> None:
    if CODEBOOK_PATH.exists():
        print(f"Reusing shared frozen codebooks: {CODEBOOK_PATH}", flush=True)
        return
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
    condition: str,
    archive: Path,
    train_annotations: Path,
    dev_annotations: Path,
    epochs: int,
    output_dir: Path,
    smoke: bool = False,
) -> Path:
    source = PROJECT / "configs/pretrain_spatial_shubert_context.json"
    if not source.exists():
        raise FileNotFoundError(f"Missing {source}; rerun the clone/install cells")
    config = json.loads(source.read_text())
    config["context_condition"] = condition
    config["objective"]["codebook_path"] = str(CODEBOOK_PATH)
    config["data"].update(
        {
            "train_archive": str(archive),
            "val_archive": str(archive),
            "train_annotations": str(train_annotations),
            "val_annotations": str(dev_annotations),
            "num_workers": 0 if smoke else 2,
            "prefetch_factor": 2,
        }
    )
    if smoke:
        config["data"].update({"limit_train_clips": 64, "limit_val_clips": 32})
    config["training"].update(
        {
            "output_dir": str(output_dir),
            "batch_size": 8 if smoke else 16,
            "epochs": epochs,
            "amp": True,
            "progress_every": 1,
            "save_every": 0,
            "display_total_epochs": 1 if smoke else len(SHARDS),
        }
    )
    destination = Path(
        f"/content/context_{condition}_{'smoke' if smoke else 'full'}.json"
    )
    destination.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return destination


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


def run_tests_and_smoke(
    archive: Path, train_annotations: Path, dev_annotations: Path
) -> None:
    state = RUN_ROOT / "smoke_complete.json"
    if state.exists():
        print("Unit tests and paired smoke training already completed.", flush=True)
        return
    print("Running context-shuffle and Spatial-SHuBERT unit tests...", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            str(PROJECT / "tests/test_context_shuffle.py"),
            str(PROJECT / "tests/test_spatial_shubert.py"),
        ],
        check=True,
    )
    smoke_root = RUN_ROOT / "smoke"
    for condition in CONDITIONS:
        output = smoke_root / condition
        config = write_config(
            condition,
            archive,
            train_annotations,
            dev_annotations,
            1,
            output,
            smoke=True,
        )
        train_once(config, None)
        if not (output / "best.pt").exists() or not (output / "metrics.csv").exists():
            raise RuntimeError(f"{condition} smoke run did not produce checkpoint/metrics")
    state.write_text(
        json.dumps(
            {
                "unit_tests": "passed",
                "paired_smoke_training": "passed",
                "train_clips_per_condition": 64,
                "val_clips_per_condition": 32,
            },
            indent=2,
        )
        + "\n"
    )


def run_diagnostics(archive: Path, annotations: Path, shard: int) -> None:
    output = DIAGNOSTIC_ROOT / f"shard_{shard:02d}"
    summary = output / "context_manipulation_summary.json"
    if summary.exists():
        print(f"Reusing context diagnostic: {summary}", flush=True)
        return
    subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "sign_semantics.context_diagnostics",
            "--archive",
            str(archive),
            "--annotations",
            str(annotations),
            "--codebooks",
            str(CODEBOOK_PATH),
            "--output-dir",
            str(output),
            "--max-frames",
            "256",
            "--block-frames",
            "16",
            "--boundary-mask-frames",
            "1",
            "--batch-size",
            "16",
            "--seed",
            "42017",
        ],
        check=True,
    )


def main() -> None:
    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    for condition in CONDITIONS:
        checkpoint_root(condition).mkdir(parents=True, exist_ok=True)
    restore_uploaded_bundle()
    state_path = RUN_ROOT / "completed_shards.json"
    completed = json.loads(state_path.read_text()) if state_path.exists() else []
    print(
        "Paired causal experiment: identical Spatial-GCN SHuBERT, objective, codebooks, "
        "seed, optimizer and exposure; real local blocks versus cross-sentence block "
        "reassignment. No linguistic labels are read.",
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
            stale = LOCAL_ROOT / f"raw_keypoints_{shard}.zip"
            if stale.exists():
                stale.unlink()
            print(f"Shard {shard} already completed; skipping.", flush=True)
            continue
        archive = LOCAL_ROOT / f"raw_keypoints_{shard}.zip"
        print(f"Paired progress: {len(completed)}/10; starting shard {shard}.", flush=True)
        print_disk(f"Before shard {shard} download")
        if shard in prefetch_jobs:
            process, log_path = prefetch_jobs.pop(shard)
            if process.wait() != 0:
                print(f"Prefetch failed ({log_path}); resuming foreground.", flush=True)
                fast_download(url(identifier), archive)
        else:
            fast_download(url(identifier), archive)
        if md5(archive) != expected_md5:
            raise RuntimeError(f"MD5 mismatch for shard {shard}")
        print_disk(f"After shard {shard} download")

        if shard == 1:
            fit_codebooks(archive, train_annotations)
            run_tests_and_smoke(archive, train_annotations, dev_annotations)
        run_diagnostics(archive, train_annotations, shard)

        future = [number for number in SHARDS if number > shard and number not in completed]
        if future and shutil.disk_usage("/content").free >= 45 * 1024**3:
            next_shard = future[0]
            next_archive = LOCAL_ROOT / f"raw_keypoints_{next_shard}.zip"
            log_path = Path(f"/content/prefetch_context_shard_{next_shard:02d}.log")
            log_handle = log_path.open("w")
            process = subprocess.Popen(
                fast_download_command(url(SHARDS[next_shard][0]), next_archive),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            log_handle.close()
            prefetch_jobs[next_shard] = (process, log_path)
            print(f"Background prefetch started for shard {next_shard}: {log_path}", flush=True)

        for condition in CONDITIONS:
            root = checkpoint_root(condition)
            config = write_config(
                condition,
                archive,
                train_annotations,
                dev_annotations,
                len(completed) + 1,
                root,
            )
            resume = root / "last.pt"
            print(
                f"=== Shard {shard}: matched condition {condition} "
                f"(global epoch {len(completed) + 1}/10) ===",
                flush=True,
            )
            train_once(config, resume if resume.exists() else None)

        completed.append(shard)
        completed_this_session += 1
        state_path.write_text(json.dumps(completed))
        make_bundle(state_path, shard)
        archive.unlink()
        elapsed = time.perf_counter() - started
        remaining = len(SHARDS) - len(completed)
        eta_hours = elapsed / max(completed_this_session, 1) * remaining / 3600
        print(
            f"Completed paired shard {shard}; local shard removed; "
            f"progress={len(completed)}/10; session ETA={eta_hours:.1f} hours.",
            flush=True,
        )
    print("All paired real/shuffled context training is complete.", flush=True)


if __name__ == "__main__":
    main()
