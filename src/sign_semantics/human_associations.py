"""Code-exact human ASL association anchors and representation RSA.

The association release identifies a prompt sign by ASL-LEX ``Sign ID``.  We
therefore treat its code, rather than an English gloss, as the analysis unit.
If ASL Citizen maps one gloss to several ASL-LEX codes, every code remains a
separate row and is never averaged with its variants.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import TruncatedSVD


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        result = list(csv.DictReader(handle))
    if not result:
        raise ValueError(f"Expected non-empty CSV: {path}")
    return result


def _truth(value: str) -> bool:
    return value.strip().lower() == "true"


def _sample_id(value: str) -> str:
    """Match converter sample IDs, which replace spaces in video stems by `_`."""
    return re.sub(r"\s+", "_", value.strip())


def _layers(payload: np.lib.npyio.NpzFile) -> list[str]:
    return sorted(key for key in payload.files if key.startswith("layer_"))


def _rdm(features: np.ndarray, metric: str = "cosine") -> np.ndarray:
    value = squareform(pdist(features, metric=metric))
    if not np.isfinite(value).all():
        raise ValueError("Non-finite RDM; a feature profile may be constant")
    return value


def _upper(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices(len(matrix), 1)]


def _code_mapping(manifest: Path) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Return gloss->codes and code->sample IDs, validating manifest identity."""
    gloss_codes: dict[str, set[str]] = defaultdict(set)
    code_samples: dict[str, set[str]] = defaultdict(set)
    for row in _rows(manifest):
        gloss, code, sample = row["id"].strip().lower(), row["asl_lex_code"].strip(), _sample_id(row["sample_id"])
        if not gloss or not code or not sample:
            raise ValueError(f"{manifest} has blank id, asl_lex_code, or sample_id")
        gloss_codes[gloss].add(code)
        code_samples[code].add(sample)
    return gloss_codes, code_samples


def build_association_anchors(
    manifest: Path,
    associations: Path,
    output_dir: Path,
    components: int = 64,
    min_responses: int = 3,
) -> None:
    """Build exact-ASL-LEX PPMI/SVD and direct-association RDMs.

    A retained response must be explicitly valid, reviewed, non-uncertain, and
    have both a response ``Tag ID`` and reviewed ``Tag Text``. Tag IDs are kept
    in the audit count; Tag Text is the shared reviewed association label used
    as the profile vocabulary (the IDs are response-video identifiers and are
    effectively unique).
    """
    gloss_codes, code_samples = _code_mapping(manifest)
    code_glosses: dict[str, set[str]] = defaultdict(set)
    for gloss, codes in gloss_codes.items():
        for code in codes:
            code_glosses[code].add(gloss)
    collisions = {code: glosses for code, glosses in code_glosses.items() if len(glosses) != 1}
    if collisions:
        raise ValueError(f"ASL-LEX code maps to multiple Citizen glosses: {sorted(collisions)[:10]}")
    candidate_codes = set(code_glosses)

    counts: dict[str, Counter[str]] = defaultdict(Counter)
    strict_tag_ids: dict[str, set[str]] = defaultdict(set)
    global_tag_counts: Counter[str] = Counter()
    direct_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in _rows(associations):
        source = row["Sign ID"].strip()
        if not (
            _truth(row["Response Valid"])
            and row["Is Tag Uncertain"].strip().lower() == "false"
            and row["Reviewer ID"].strip()
            and row["Tag ID"].strip()
            and row["Tag Text"].strip()
        ):
            continue
        tag_id, tag = row["Tag ID"].strip(), row["Tag Text"].strip()
        # Repeated Tag IDs would be duplicate records rather than independent
        # responses. Retain one exact reviewed response token per source.
        if tag_id in strict_tag_ids[source]:
            continue
        strict_tag_ids[source].add(tag_id)
        global_tag_counts[tag] += 1
        if source not in candidate_codes:
            continue
        counts[source][tag] += 1
        if tag in candidate_codes:
            direct_counts[source][tag] += 1

    usable = sorted(code for code in candidate_codes if len(strict_tag_ids[code]) >= min_responses)
    if len(usable) < 3:
        raise ValueError(f"Only {len(usable)} exact-code signs meet min_responses={min_responses}")
    vocabulary = sorted({tag for code in usable for tag in counts[code]})
    vocabulary_index = {tag: index for index, tag in enumerate(vocabulary)}
    count_matrix = np.zeros((len(usable), len(vocabulary)), dtype=np.float64)
    for row, code in enumerate(usable):
        for tag, value in counts[code].items():
            count_matrix[row, vocabulary_index[tag]] = value
    total = float(sum(global_tag_counts.values()))
    row_totals = count_matrix.sum(axis=1, keepdims=True)
    column_totals = np.asarray([[global_tag_counts[tag] for tag in vocabulary]], dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ppmi = np.maximum(np.log((count_matrix * total) / (row_totals * column_totals)), 0.0)
    ppmi[~np.isfinite(ppmi)] = 0.0
    usable_components = min(components, len(usable) - 1, len(vocabulary) - 1)
    if usable_components < 1:
        raise ValueError("Association profile has insufficient rank for SVD")
    svd = TruncatedSVD(n_components=usable_components, n_iter=10, random_state=42)
    svd_features = svd.fit_transform(ppmi)

    selected_index = {code: index for index, code in enumerate(usable)}
    direct = np.zeros((len(usable), len(usable)), dtype=np.float64)
    for row, source in enumerate(usable):
        for target, value in direct_counts[source].items():
            if target in selected_index:
                direct[row, selected_index[target]] = value
    direct_profile = np.log1p(direct)
    profile_rdm = _rdm(ppmi)
    svd_rdm = _rdm(svd_features)
    direct_usable = np.flatnonzero(np.linalg.norm(direct_profile, axis=1) > 0)
    direct_ids = np.asarray([usable[index] for index in direct_usable])
    direct_rdm = (
        _rdm(direct_profile[direct_usable])
        if len(direct_usable) >= 3
        else np.empty((0, 0), dtype=np.float64)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "human_association_profiles.npz",
        ids=np.asarray(usable),
        features=svd_features,
        ppmi=ppmi,
        vocabulary=np.asarray(vocabulary),
        direct_profile=direct_profile,
        direct_ids=direct_ids,
        components=usable_components,
        min_responses=min_responses,
        strict_filter=(
            "Response_Valid=true; Reviewer_ID_present; Is_Tag_Uncertain=false; "
            "Tag_ID_and_reviewed_Tag_Text_present; de-duplicate_Tag_ID_within_source"
        ),
        interpretation=(
            "ASL-LEX-code-exact human association profiles; PPMI tag baseline uses all strict reviewed prompts; multi-code English glosses remain separate"
        ),
    )
    for name, ids, rdm, source in (
        ("human_association_ppmi_rdm.npz", usable, profile_rdm, "PPMI association profile cosine"),
        ("human_association_svd_rdm.npz", usable, svd_rdm, "PPMI truncated-SVD cosine"),
        ("human_association_direct_rdm.npz", direct_ids, direct_rdm, "direct reviewed code-response profile cosine"),
    ):
        np.savez_compressed(
            output_dir / name,
            ids=np.asarray(ids),
            rdm=rdm,
            metric="cosine",
            source=source,
            interpretation="exact ASL-LEX code units; direct RDM is a robustness analysis",
        )
    audit_fields = (
        "concept_id", "asl_lex_code", "citizen_token_count", "gloss_code_count",
        "variant_status", "strict_reviewed_responses", "profile_tags", "direct_code_responses",
        "profile_included", "direct_rdm_included",
    )
    audit_rows = []
    for code in sorted(candidate_codes):
        gloss = next(iter(code_glosses[code]))
        audit_rows.append(
            {
                "concept_id": gloss,
                "asl_lex_code": code,
                "citizen_token_count": str(len(code_samples[code])),
                "gloss_code_count": str(len(gloss_codes[gloss])),
                "variant_status": "single_code_gloss" if len(gloss_codes[gloss]) == 1 else "multiple_codes_kept_separate",
                "strict_reviewed_responses": str(len(strict_tag_ids[code])),
                "profile_tags": str(len(counts[code])),
                "direct_code_responses": str(sum(direct_counts[code].values())),
                "profile_included": str(code in selected_index).lower(),
                "direct_rdm_included": str(code in set(direct_ids)).lower(),
            }
        )
    with (output_dir / "human_association_code_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit_fields)
        writer.writeheader()
        writer.writerows(audit_rows)
    print(
        f"Saved exact-code human association anchors for {len(usable)}/{len(candidate_codes)} "
        f"ASL Citizen code groups ({len(direct_ids)} direct-association robustness units) to {output_dir}"
    )


def aggregate_model_tokens_by_code(tokens: Path, manifest: Path, output: Path) -> None:
    """Average only model tokens belonging to the same exact ASL-LEX code."""
    _, code_samples = _code_mapping(manifest)
    sample_code = {sample: code for code, samples in code_samples.items() for sample in samples}
    with np.load(tokens, allow_pickle=False) as payload:
        if "sample_ids" not in payload or "ids" not in payload:
            raise ValueError(f"{tokens} must contain ids and sample_ids")
        sample_ids = [_sample_id(str(value)) for value in payload["sample_ids"]]
        layers = _layers(payload)
        if not layers:
            raise ValueError(f"{tokens} has no layer_* arrays")
        unknown = sorted(set(sample_ids) - set(sample_code))
        if unknown:
            raise ValueError(f"{tokens} contains sample IDs absent from manifest: {unknown[:10]}")
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, sample in enumerate(sample_ids):
            grouped[sample_code[sample]].append(index)
        codes = sorted(grouped)
        arrays = {key: np.stack([payload[key][grouped[code]].mean(axis=0) for code in codes]) for key in layers}
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        ids=np.asarray(codes), counts=np.asarray([len(grouped[code]) for code in codes]),
        source=str(tokens), interpretation="model tokens averaged within exact ASL-LEX code only", **arrays,
    )
    print(f"Saved {len(codes)} exact-code model representations to {output}")


def restrict_gloss_features_to_single_codes(features: Path, manifest: Path, output: Path) -> None:
    """Map gloss representations only where one Citizen gloss has one ASL-LEX code.

    BERT has an English-gloss representation, not a sign-variant representation;
    duplicated code rows would artificially create zero BERT distances. This
    conservative mapping excludes multi-code glosses instead of duplicating.
    """
    gloss_codes, _ = _code_mapping(manifest)
    with np.load(features, allow_pickle=False) as payload:
        ids = [str(value).strip().lower() for value in payload["ids"]]
        lookup = {identifier: index for index, identifier in enumerate(ids)}
        layers = _layers(payload)
        selected = sorted((next(iter(codes)), gloss) for gloss, codes in gloss_codes.items() if len(codes) == 1 and gloss in lookup)
        if len(selected) < 3:
            raise ValueError("Fewer than three single-code gloss representations are available")
        codes, glosses = zip(*selected)
        arrays = {key: payload[key][[lookup[gloss] for gloss in glosses]] for key in layers}
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, ids=np.asarray(codes), gloss_ids=np.asarray(glosses), source=str(features),
        interpretation="gloss features restricted to one-ASL-LEX-code concepts; variants excluded", **arrays,
    )
    print(f"Saved {len(codes)} single-code gloss representations to {output}")


def code_features_to_single_code_glosses(
    features: Path, manifest: Path, output: Path, key: str = "features"
) -> None:
    """Map exact-code human features to gloss IDs only for one-code glosses."""
    gloss_codes, _ = _code_mapping(manifest)
    with np.load(features, allow_pickle=False) as payload:
        codes = [str(value).strip() for value in payload["ids"]]
        if key not in payload:
            raise ValueError(f"{features} does not contain {key!r}")
        values = np.asarray(payload[key], dtype=np.float64)
        if values.ndim != 2 or len(values) != len(codes):
            raise ValueError(f"{features}:{key} must be a code-aligned 2D matrix")
        lookup = {code: index for index, code in enumerate(codes)}
        selected = sorted(
            (gloss, next(iter(code_set)))
            for gloss, code_set in gloss_codes.items()
            if len(code_set) == 1 and next(iter(code_set)) in lookup
        )
        arrays = np.stack([values[lookup[code]] for _, code in selected])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        ids=np.asarray([gloss for gloss, _ in selected]),
        asl_lex_codes=np.asarray([code for _, code in selected]),
        features=arrays,
        source=str(features),
        source_key=key,
        interpretation=(
            "concept-level single-code gloss mapping; EEG stimulus code variant remains unknown"
        ),
    )
    print(f"Saved {len(selected)} single-code gloss human feature rows to {output}")


def build_iconicity_gloss_features(asl_lex: Path, manifest: Path, output_dir: Path) -> None:
    """Export Deaf-primary and hearing-rater iconicity for unambiguous glosses."""
    gloss_codes, _ = _code_mapping(manifest)
    rows = _rows_with_encoding(asl_lex, "latin-1")
    by_code = {row["Code"].strip(): row for row in rows if row.get("Code", "").strip()}
    output_dir.mkdir(parents=True, exist_ok=True)
    for column, filename, interpretation in (
        ("D.Iconicity(M)", "deaf_iconicity_single_code_gloss.npz", "Deaf ASL signer iconicity mean"),
        ("Iconicity(M)", "hearing_iconicity_single_code_gloss.npz", "hearing non-signer iconicity mean"),
    ):
        selected: list[tuple[str, str, float]] = []
        for gloss, codes in gloss_codes.items():
            if len(codes) != 1:
                continue
            code = next(iter(codes))
            raw = by_code.get(code, {}).get(column, "").strip()
            try:
                value = float(raw)
            except ValueError:
                continue
            if np.isfinite(value):
                selected.append((gloss, code, value))
        selected.sort()
        np.savez_compressed(
            output_dir / filename,
            ids=np.asarray([row[0] for row in selected]),
            asl_lex_codes=np.asarray([row[1] for row in selected]),
            features=np.asarray([[row[2]] for row in selected]),
            column=column,
            source=str(asl_lex),
            interpretation=interpretation + "; single-code ASL Citizen gloss mapping",
        )
        print(f"Saved {len(selected)} {interpretation} rows to {output_dir / filename}")
        # Exact-code version retains variants for model-space controls.
        candidate_codes = sorted({code for codes in gloss_codes.values() for code in codes})
        exact = []
        for code in candidate_codes:
            raw = by_code.get(code, {}).get(column, "").strip()
            try:
                value = float(raw)
            except ValueError:
                continue
            if np.isfinite(value):
                exact.append((code, value))
        exact_filename = filename.replace("single_code_gloss", "exact_code")
        np.savez_compressed(
            output_dir / exact_filename,
            ids=np.asarray([row[0] for row in exact]),
            features=np.asarray([[row[1]] for row in exact]),
            column=column,
            source=str(asl_lex),
            interpretation=interpretation + "; exact ASL-LEX code units",
        )
        print(f"Saved {len(exact)} exact-code {interpretation} rows to {output_dir / exact_filename}")


def _rows_with_encoding(path: Path, encoding: str) -> list[dict[str, str]]:
    with path.open(encoding=encoding, newline="") as handle:
        result = list(csv.DictReader(handle))
    if not result:
        raise ValueError(f"Expected non-empty CSV: {path}")
    return result


def run_human_representation_rsa(
    human_rdm: Path,
    representation: Path,
    output: Path,
    metric: str = "cosine",
    permutations: int = 0,
    seed: int = 42,
) -> None:
    """Layerwise exact-code human-RDM RSA; permutations are optional and bounded by caller."""
    if permutations < 0:
        raise ValueError("permutations must be non-negative")
    with np.load(human_rdm, allow_pickle=False) as human, np.load(representation, allow_pickle=False) as model:
        human_ids = [str(value).strip().lower() for value in human["ids"]]
        human_matrix = np.asarray(human["rdm"], dtype=float)
        model_ids = [str(value).strip().lower() for value in model["ids"]]
        common = sorted(set(human_ids) & set(model_ids))
        if len(common) < 3:
            raise ValueError("Human RDM and representation share fewer than three exact ASL-LEX codes")
        h_index, m_index = [human_ids.index(code) for code in common], [model_ids.index(code) for code in common]
        human_vector = _upper(human_matrix[np.ix_(h_index, h_index)])
        layers = _layers(model)
        if not layers:
            raise ValueError(f"{representation} has no layer_* arrays")
        model_vectors = np.stack([_upper(_rdm(model[key][m_index], metric)) for key in layers])
    observed = np.asarray([spearmanr(human_vector, vector).statistic for vector in model_vectors])
    if not np.isfinite(observed).all():
        raise ValueError("Human or representation RDM is constant")
    p = np.full(len(layers), np.nan)
    p_fwer = np.full(len(layers), np.nan)
    if permutations:
        rng = np.random.default_rng(seed)
        raw = np.zeros(len(layers), dtype=int)
        maximum = np.zeros(len(layers), dtype=int)
        # A concept-label permutation only reorders the already ranked RDM
        # entries.  Rank once, then compute Pearson correlations of centered
        # ranks.  This is exactly Spearman RSA but avoids sorting every RDM at
        # every permutation (the formal 10k run would otherwise be needlessly
        # expensive).
        triangle = np.triu_indices(len(common), 1)
        human_ranks = rankdata(human_vector)
        human_ranks -= human_ranks.mean()
        human_rank_matrix = np.zeros((len(common), len(common)), dtype=np.float64)
        human_rank_matrix[triangle] = human_ranks
        human_rank_matrix[(triangle[1], triangle[0])] = human_ranks
        model_ranks = np.stack([rankdata(vector) for vector in model_vectors])
        model_ranks -= model_ranks.mean(axis=1, keepdims=True)
        model_norm = np.linalg.norm(model_ranks, axis=1)
        human_norm = np.linalg.norm(human_ranks)
        for _ in range(permutations):
            order = rng.permutation(len(common))
            permuted_ranks = _upper(human_rank_matrix[np.ix_(order, order)])
            null = (model_ranks @ permuted_ranks) / (model_norm * human_norm)
            raw += np.abs(null) >= np.abs(observed)
            maximum += np.max(np.abs(null)) >= np.abs(observed)
        p, p_fwer = (raw + 1) / (permutations + 1), (maximum + 1) / (permutations + 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("representation_layer", "n_codes", "spearman_rho", "permutation_p", "permutation_p_fwer", "metric", "permutations"))
        writer.writeheader()
        for index, layer in enumerate(layers):
            writer.writerow({"representation_layer": layer, "n_codes": len(common), "spearman_rho": observed[index], "permutation_p": p[index], "permutation_p_fwer": p_fwer[index], "metric": metric, "permutations": permutations})
    print(f"Saved human-RDM RSA for {len(layers)} layers and {len(common)} exact codes to {output}")


def main_build() -> None:
    parser = argparse.ArgumentParser(description="Build exact-ASL-LEX human association anchors")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--associations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--components", type=int, default=64)
    parser.add_argument("--min-responses", type=int, default=3)
    args = parser.parse_args()
    build_association_anchors(**vars(args))


def main_model_codes() -> None:
    parser = argparse.ArgumentParser(description="Aggregate token model embeddings by exact ASL-LEX code")
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    aggregate_model_tokens_by_code(args.tokens, args.manifest, args.output)


def main_single_code_glosses() -> None:
    parser = argparse.ArgumentParser(description="Restrict gloss features to unambiguous ASL-LEX codes")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    restrict_gloss_features_to_single_codes(args.features, args.manifest, args.output)


def main_rsa() -> None:
    parser = argparse.ArgumentParser(description="Layerwise human-association to representation RSA")
    parser.add_argument("--human-rdm", type=Path, required=True)
    parser.add_argument("--representation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metric", choices=("cosine", "correlation"), default="cosine")
    parser.add_argument("--permutations", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_human_representation_rsa(**vars(args))


def main_code_to_gloss() -> None:
    parser = argparse.ArgumentParser(description="Map exact-code features to single-code glosses")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key", default="features")
    args = parser.parse_args()
    code_features_to_single_code_glosses(**vars(args))


def main_iconicity() -> None:
    parser = argparse.ArgumentParser(description="Build ASL-LEX iconicity feature blocks")
    parser.add_argument("--asl-lex", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build_iconicity_gloss_features(**vars(args))
