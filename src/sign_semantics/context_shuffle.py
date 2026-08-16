from __future__ import annotations

"""Matched real/shuffled context batches for continuous skeleton pretraining.

The frozen Spatial-SHuBERT targets are four independent *frame-level* cluster
labels, not an established unified span tokenizer.  Treating every change in
their four-way Cartesian product as a unit boundary would usually degenerate
into frame shuffling.  We therefore use fixed-duration local trajectory blocks
as the closest defensible span-preserving manipulation: every frame inside a
block remains intact and ordered, while complete blocks are reassigned across
sentences.

Both conditions use the same block layout and hide the same small region on
each side of every join from Transformer attention.  This prevents the
shuffled condition from gaining extra supervised targets exactly at artificial
joins.  Because that boundary control changes the effective attention mask,
the causal comparison requires a newly trained real-context replication; the
historical real-context checkpoint remains a reference only.
"""

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import Sampler, default_collate

from .features import STREAM_JOINTS
from .skeleton_codebooks import PARTS, SkeletonCodebooks, frame_descriptors, split_parts


CONTEXT_CONDITIONS = ("real", "shuffled")
FORBIDDEN_TRAINING_KEYS = (
    "asl_citizen",
    "asl_lex",
    "bert",
    "deaf",
    "eeg",
    "english_translation",
    "gloss",
    "iconicity",
    "semantic_association",
)


def assert_no_linguistic_supervision(config: dict) -> None:
    """Reject linguistic/evaluation resources as config fields.

    The official structural annotation files are historically named
    ``YT.translations.*``.  Their *values* are never consumed: data.py reads
    only top-level source IDs and ``clip_order``/clip keys.  For that reason we
    audit config key names rather than rejecting those unavoidable filenames.
    """

    def visit(value: object, trail: tuple[str, ...] = ()) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                normalized = str(key).strip().lower()
                if any(token in normalized for token in FORBIDDEN_TRAINING_KEYS):
                    raise ValueError(
                        "Linguistic/evaluation supervision is forbidden in training config: "
                        + ".".join((*trail, str(key)))
                    )
                visit(child, (*trail, str(key)))
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, (*trail, str(index)))
        elif isinstance(value, str):
            normalized = value.strip().lower()
            # SHuBERT is the name of the skeleton model, not a text-BERT input.
            normalized_for_audit = normalized.replace("shubert", "")
            annotation_path = bool(trail) and trail[-1].endswith("annotations")
            value_tokens = (*FORBIDDEN_TRAINING_KEYS, "translation")
            matches = [token for token in value_tokens if token in normalized_for_audit]
            if matches and not (
                annotation_path and set(matches).issubset({"translation"})
            ):
                raise ValueError(
                    "Linguistic/evaluation resource path is forbidden in training config: "
                    + ".".join(trail)
                )

    visit(config)


class SourceGroupedBatchSampler(Sampler[list[int]]):
    """Deterministic batches that keep source-video clips nearby where possible."""

    def __init__(
        self,
        source_groups: Sequence[str],
        batch_size: int,
        seed: int,
        shuffle: bool,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not source_groups:
            raise ValueError("source_groups cannot be empty")
        self.source_groups = [str(value) for value in source_groups]
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return math.ceil(len(self.source_groups) / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, group in enumerate(self.source_groups):
            grouped[group].append(index)
        generator = np.random.default_rng(self.seed + self.epoch)
        group_names = sorted(grouped)
        if self.shuffle:
            generator.shuffle(group_names)
        flattened: list[int] = []
        for group in group_names:
            indices = list(grouped[group])
            if self.shuffle:
                generator.shuffle(indices)
            flattened.extend(indices)
        for start in range(0, len(flattened), self.batch_size):
            yield flattened[start : start + self.batch_size]


@dataclass(frozen=True)
class LocalBlock:
    row: int
    block: int
    start: int
    stop: int
    clip_id: str
    source_group: str

    @property
    def duration(self) -> int:
        return self.stop - self.start


def local_blocks(
    ids: Sequence[str], source_groups: Sequence[str], lengths: Sequence[int], block_frames: int
) -> list[LocalBlock]:
    if block_frames < 2:
        raise ValueError("block_frames must be at least two")
    blocks: list[LocalBlock] = []
    for row, (clip_id, source, length) in enumerate(
        zip(ids, source_groups, lengths, strict=True)
    ):
        if length < 1:
            continue
        for block, start in enumerate(range(0, int(length), block_frames)):
            blocks.append(
                LocalBlock(
                    row=row,
                    block=block,
                    start=start,
                    stop=min(start + block_frames, int(length)),
                    clip_id=str(clip_id),
                    source_group=str(source),
                )
            )
    return blocks


def _batch_seed(seed: int, ids: Sequence[str]) -> int:
    payload = f"{seed}:" + "|".join(sorted(str(value) for value in ids))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def assign_donor_blocks(
    blocks: Sequence[LocalBlock],
    seed: int,
    transition_cost: np.ndarray | None = None,
) -> tuple[list[int], dict[str, float]]:
    """Find a duration-exact block permutation with cross-sentence/source preference."""
    assignment = list(range(len(blocks)))
    generator = np.random.default_rng(seed)
    for duration in sorted({block.duration for block in blocks}):
        positions = [index for index, block in enumerate(blocks) if block.duration == duration]
        if len(positions) < 2:
            continue
        size = len(positions)
        cost = generator.random((size, size)) * 1e-3
        for destination_local, destination_index in enumerate(positions):
            destination = blocks[destination_index]
            for donor_local, donor_index in enumerate(positions):
                donor = blocks[donor_index]
                if donor.clip_id == destination.clip_id:
                    # Effectively a hard constraint whenever a duration-matched
                    # donor from another sentence exists.
                    cost[destination_local, donor_local] += 1_000_000_000.0
                elif donor.source_group != destination.source_group:
                    cost[destination_local, donor_local] += 10.0
                if transition_cost is not None:
                    cost[destination_local, donor_local] += float(
                        transition_cost[destination_index, donor_index]
                    ) * 100.0
        rows, columns = linear_sum_assignment(cost)
        for row, column in zip(rows, columns, strict=True):
            assignment[positions[int(row)]] = positions[int(column)]
    cross_sentence = np.asarray(
        [blocks[destination].clip_id != blocks[donor].clip_id for destination, donor in enumerate(assignment)]
    )
    same_source = np.asarray(
        [blocks[destination].source_group == blocks[donor].source_group for destination, donor in enumerate(assignment)]
    )
    moved = np.asarray([destination != donor for destination, donor in enumerate(assignment)])
    return assignment, {
        "blocks": float(len(blocks)),
        "moved_block_fraction": float(moved.mean()) if len(moved) else 0.0,
        "cross_sentence_fraction": float(cross_sentence.mean()) if len(cross_sentence) else 0.0,
        "same_source_fraction": float(same_source.mean()) if len(same_source) else 0.0,
    }


def _cross_boundary_jump(
    streams: dict[str, torch.Tensor],
    observed: dict[str, torch.Tensor],
    left_row: int,
    left_frame: int,
    right_row: int,
    right_frame: int,
) -> float:
    distances = []
    for name in STREAM_JOINTS:
        usable = (
            observed[name][left_row, left_frame]
            & observed[name][right_row, right_frame]
        )
        if usable.any():
            distances.append(
                torch.linalg.vector_norm(
                    streams[name][left_row, left_frame, usable]
                    - streams[name][right_row, right_frame, usable],
                    dim=-1,
                )
            )
    if not distances:
        return 0.0
    return float(torch.cat(distances).mean())


def _boundary_jump(
    streams: dict[str, torch.Tensor],
    observed: dict[str, torch.Tensor],
    row: int,
    left: int,
    right: int,
) -> float:
    return _cross_boundary_jump(streams, observed, row, left, row, right)


def _frame_feature(
    streams: dict[str, torch.Tensor], row: int, frame: int
) -> torch.Tensor:
    return torch.cat(
        [streams[name][row, frame].reshape(-1).float() for name in STREAM_JOINTS]
    )


class ContextBatchCollator:
    """Collate real or block-reassigned skeleton sentences with matched joins."""

    def __init__(
        self,
        condition: str,
        block_frames: int,
        boundary_mask_frames: int,
        seed: int,
        codebooks: SkeletonCodebooks | None = None,
        include_detailed_mapping: bool = False,
    ) -> None:
        if condition not in CONTEXT_CONDITIONS:
            raise ValueError(f"condition must be one of {CONTEXT_CONDITIONS}")
        if block_frames < 2:
            raise ValueError("block_frames must be at least two")
        if boundary_mask_frames < 0 or boundary_mask_frames * 2 >= block_frames:
            raise ValueError("boundary_mask_frames must be non-negative and less than half a block")
        self.condition = condition
        self.block_frames = int(block_frames)
        self.boundary_mask_frames = int(boundary_mask_frames)
        self.seed = int(seed)
        self.codebooks = codebooks
        self.include_detailed_mapping = bool(
            include_detailed_mapping or codebooks is not None
        )

    def __call__(self, items: Sequence[dict]) -> dict:
        batch = default_collate(items)
        ids = [str(value) for value in batch["id"]]
        sources = [str(value) for value in batch["source_group"]]
        original_valid = batch["valid"].clone()
        lengths = [int(row.sum()) for row in original_valid]
        blocks = local_blocks(ids, sources, lengths, self.block_frames)
        original_streams = {name: batch[name].clone() for name in STREAM_JOINTS}
        original_observed = {
            name: batch[f"{name}_observed"].clone() for name in STREAM_JOINTS
        }
        if self.condition == "shuffled":
            # Vectorized boundary matching. Immediate join frames are hidden
            # below; costs use the first visible frame on each side.
            previous_features = []
            start_features = []
            has_previous = []
            for block in blocks:
                left = max(0, block.start - self.boundary_mask_frames - 1)
                right = min(block.stop - 1, block.start + self.boundary_mask_frames)
                previous_features.append(
                    _frame_feature(original_streams, block.row, left)
                )
                start_features.append(
                    _frame_feature(original_streams, block.row, right)
                )
                has_previous.append(block.block > 0)
            previous_matrix = torch.stack(previous_features)
            start_matrix = torch.stack(start_features)
            candidate_jump = torch.cdist(previous_matrix, start_matrix)
            target_jump = torch.linalg.vector_norm(
                previous_matrix - start_matrix, dim=1
            )
            transition_cost = torch.abs(candidate_jump - target_jump[:, None])
            transition_cost[~torch.tensor(has_previous, dtype=torch.bool)] = 0
            assignment, audit = assign_donor_blocks(
                blocks,
                _batch_seed(self.seed, ids),
                transition_cost.numpy(),
            )
        else:
            assignment = list(range(len(blocks)))
            audit = {
                "blocks": float(len(blocks)),
                "moved_block_fraction": 0.0,
                "cross_sentence_fraction": 0.0,
                "same_source_fraction": 1.0,
            }

        if self.condition == "shuffled":
            for destination_index, donor_index in enumerate(assignment):
                destination, donor = blocks[destination_index], blocks[donor_index]
                if destination.duration != donor.duration:
                    raise AssertionError("Block reassignment changed duration")
                destination_slice = slice(destination.start, destination.stop)
                donor_slice = slice(donor.start, donor.stop)
                for name in STREAM_JOINTS:
                    batch[name][destination.row, destination_slice] = original_streams[name][
                        donor.row, donor_slice
                    ]
                    batch[f"{name}_observed"][destination.row, destination_slice] = (
                        original_observed[name][donor.row, donor_slice]
                    )

        real_jumps = []
        assigned_jumps = []
        real_visible_jumps = []
        assigned_visible_jumps = []
        assigned_streams = {name: batch[name] for name in STREAM_JOINTS}
        assigned_observed = {
            name: batch[f"{name}_observed"] for name in STREAM_JOINTS
        }
        for row, length in enumerate(lengths):
            for join in range(self.block_frames, length, self.block_frames):
                real_jumps.append(
                    _boundary_jump(
                        original_streams, original_observed, row, join - 1, join
                    )
                )
                assigned_jumps.append(
                    _boundary_jump(
                        assigned_streams, assigned_observed, row, join - 1, join
                    )
                )
                visible_left = max(
                    0, join - self.boundary_mask_frames - 1
                )
                visible_right = min(
                    length - 1, join + self.boundary_mask_frames
                )
                real_visible_jumps.append(
                    _boundary_jump(
                        original_streams,
                        original_observed,
                        row,
                        visible_left,
                        visible_right,
                    )
                )
                assigned_visible_jumps.append(
                    _boundary_jump(
                        assigned_streams,
                        assigned_observed,
                        row,
                        visible_left,
                        visible_right,
                    )
                )

        # Identical blocked positions are removed from Transformer attention and
        # from masked-cluster supervision in both causal conditions.
        boundary_mask = torch.zeros_like(original_valid)
        if self.boundary_mask_frames:
            for row, length in enumerate(lengths):
                for join in range(self.block_frames, length, self.block_frames):
                    start = max(0, join - self.boundary_mask_frames)
                    stop = min(length, join + self.boundary_mask_frames)
                    boundary_mask[row, start:stop] = True
        batch["valid"] = original_valid & ~boundary_mask
        batch["context_boundary_mask"] = boundary_mask
        batch["context_original_valid"] = original_valid
        batch["context_original_lengths"] = torch.tensor(lengths, dtype=torch.long)
        audit["join_count"] = float(len(real_jumps))
        audit["real_boundary_jump_sum"] = float(sum(real_jumps))
        audit["assigned_boundary_jump_sum"] = float(sum(assigned_jumps))
        audit["real_visible_boundary_jump_sum"] = float(sum(real_visible_jumps))
        audit["assigned_visible_boundary_jump_sum"] = float(
            sum(assigned_visible_jumps)
        )
        audit["masked_boundary_frames"] = float(boundary_mask.sum())
        audit["original_valid_frames"] = float(original_valid.sum())

        signatures: list[dict[str, int]] | None = None
        if self.codebooks is not None:
            parts, _ = split_parts(original_streams, original_observed)
            # Vectorize the diagnostic target lookup: one cdist per part and
            # batch, rather than one cdist per block. This leaves the statistic
            # unchanged but avoids thousands of tiny CPU kernels per batch.
            part_units: dict[str, list[int]] = {}
            for name, values in parts.items():
                descriptors = torch.stack(
                    [
                        frame_descriptors(
                            values[block.row, block.start : block.stop].mean(dim=0)
                        )
                        for block in blocks
                    ]
                )
                part_units[name] = [
                    int(value)
                    for value in self.codebooks.targets(name, descriptors)
                ]
            signatures = [
                {name: part_units[name][index] for name in PARTS}
                for index in range(len(blocks))
            ]

        if self.include_detailed_mapping:
            mapping = []
            for destination_index, donor_index in enumerate(assignment):
                destination, donor = blocks[destination_index], blocks[donor_index]
                record: dict[str, object] = {
                    "destination_clip": destination.clip_id,
                    "destination_source": destination.source_group,
                    "destination_block": destination.block,
                    "destination_start": destination.start,
                    "destination_stop": destination.stop,
                    "donor_clip": donor.clip_id,
                    "donor_source": donor.source_group,
                    "donor_block": donor.block,
                    "donor_start": donor.start,
                    "donor_stop": donor.stop,
                }
                if signatures is not None:
                    record["real_units"] = signatures[destination_index]
                    record["assigned_units"] = signatures[donor_index]
                mapping.append(record)
            batch["context_mapping"] = mapping
        batch["context_assignment"] = assignment
        batch["context_batch_audit"] = audit
        return batch


def source_groups_for_dataset(dataset: object) -> list[str]:
    members = getattr(dataset, "members", None)
    mapping = getattr(dataset, "source_groups", None)
    if members is None or mapping is None:
        raise TypeError("Context batching requires YouTubeASLPoseDataset metadata")
    from pathlib import PurePosixPath

    return [mapping[PurePosixPath(member).stem] for member in members]
