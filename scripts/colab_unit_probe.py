"""One-click Colab pilot for recurring continuous-sign unit discovery."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

from google.colab import files


BASE = "https://lindat.mff.cuni.cz/repository/server/api/core/bitstreams"
SHARD_ID = "3dc57bf4-c5fb-491c-8ab2-a9215e0e2fe5"
SHARD_MD5 = "75223dd5e7e9b6ccb9f34c5792fc1e6d"
TRAIN_ANNOTATION_ID = "f8460818-3605-4f05-9832-90ddc68f22e6"
PROJECT = Path("/content/youtube-asl-skeleton-bert")
INPUT_BUNDLE = Path("/content/sign_unit_probe_inputs.zip")
WORK = Path("/content/sign_unit_probe")
OUTPUT = WORK / "results"


def url(identifier: str) -> str:
    return f"{BASE}/{identifier}/content"


def download(source: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "aria2c",
            "--continue=true",
            "--max-connection-per-server=16",
            "--split=16",
            "--min-split-size=16M",
            "--file-allocation=none",
            "--auto-file-renaming=false",
            "--max-tries=0",
            "--retry-wait=5",
            "--summary-interval=10",
            f"--dir={target.parent}",
            f"--out={target.name}",
            source,
        ],
        check=True,
    )


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if not PROJECT.exists():
        raise FileNotFoundError(f"Project not cloned: {PROJECT}")
    if not INPUT_BUNDLE.exists():
        raise FileNotFoundError(
            f"Upload the prepared local bundle as {INPUT_BUNDLE.name} before this cell"
        )
    with zipfile.ZipFile(INPUT_BUNDLE) as handle:
        handle.extractall(PROJECT)
    checkpoint = PROJECT / "data/checkpoints/contrastive/best.pt"
    manifest = PROJECT / "data/processed/asl_citizen_neural_subset/lexical_evaluation_manifest.csv"
    if not checkpoint.exists() or not manifest.exists():
        raise RuntimeError("Input bundle lacks the contrastive checkpoint or lexical manifest")

    WORK.mkdir(parents=True, exist_ok=True)
    annotations = WORK / "YT.translations.train.json"
    archive = WORK / "raw_keypoints_1.zip"
    if not annotations.exists():
        download(url(TRAIN_ANNOTATION_ID), annotations)
    if not archive.exists():
        download(url(SHARD_ID), archive)
    if md5(archive) != SHARD_MD5:
        raise RuntimeError("Shard 1 MD5 mismatch; keeping partial file for diagnosis/resume")

    command = [
        sys.executable,
        "-u",
        "-m",
        "sign_semantics.unit_probe",
        "--checkpoint",
        str(checkpoint),
        "--archive",
        str(archive),
        "--annotations",
        str(annotations),
        "--lexical-manifest",
        str(manifest),
        "--root",
        str(PROJECT),
        "--output-dir",
        str(OUTPUT),
        "--sample-clips",
        "5000",
        "--layer",
        "1",
        "--patch-frames",
        "8",
        "--patch-stride",
        "4",
        "--clusters",
        "100",
        "--batch-size",
        "32",
        "--permutations",
        "1000",
        "--seed",
        "42",
    ]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    gate = json.loads((OUTPUT / "pilot_gate.json").read_text())
    print("Pilot gate (exploratory thresholds):", json.dumps(gate, indent=2), flush=True)

    bundle = Path("/content/sign_unit_probe_results.zip")
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in OUTPUT.rglob("*"):
            if path.is_file():
                handle.write(path, path.relative_to(OUTPUT.parent))
    if bundle.stat().st_size == 0:
        raise RuntimeError("Refusing to delete data before result bundle verification")
    archive.unlink()
    print("Raw shard deleted only after the result bundle was verified.", flush=True)
    files.download(str(bundle))


if __name__ == "__main__":
    main()
