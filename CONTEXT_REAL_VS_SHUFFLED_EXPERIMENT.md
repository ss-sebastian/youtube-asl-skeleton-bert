# Real versus shuffled continuous-ASL context experiment

This experiment tests only whether natural sentence context contributes to lexical-semantic geometry when visual content, architecture, cluster vocabulary, masking, optimization, seed, and exposure are matched. Training uses no gloss, translation text, ASL Citizen/ASL-LEX labels, human ratings, BERT, iconicity, or EEG information.

## Audit of the existing Spatial-SHuBERT pipeline

1. **Skeleton loading.** `YouTubeASLPoseDataset` reads sentence JSONs directly from one shard ZIP. Structural annotation files provide allowed clip IDs. Translation values are ignored. Each frame retains 25 body, 42 hand, and 37 selected face landmarks in 2D. Coordinates are normalized using the existing shoulder-centred/shoulder-scaled pipeline. Clips longer than 256 frames use a random contiguous crop for training and deterministic linear resampling for validation; shorter clips are padded.
2. **Cluster targets.** Four frozen `MiniBatchKMeans` codebooks are fit once on at most 100,000 sufficiently observed frames from shard 1: body, right hand, left hand, and face. Each part has 100 clusters. Inputs are standardized flattened frame-local normalized coordinates.
3. **Target temporal level.** Targets are frame-level. There is no existing unified pseudo-unit vocabulary or validated pseudo-unit span segmentation.
4. **Masking.** Approximately 40% of valid frames are selected in variable contiguous spans centred around 20 frames. Padding and, in the new paired experiment, matched block-boundary gaps are never masked or supervised.
5. **Prediction.** A separate linear head predicts one of 100 frozen cluster IDs for each of the four parts at masked, sufficiently observed frames. Cross-entropies are averaged across usable parts. Validation reports total and part-specific losses/accuracies.
6. **Context window.** The six-layer bidirectional BERT receives the complete cropped/resampled sentence up to 256 positions. The Spatial GCN is frame-local; BERT supplies all temporal/contextual interaction.
7. **Identity metadata.** The original training model uses no signer/video embedding. The top-level annotation key maps clips to a source video and is available as non-linguistic grouping metadata. It is not validated as a unique signer ID and must be reported as `source_video`, not `signer`.
8. **Split.** Official train/dev structural inventories select clips. Each shard is used sequentially for one global stage. Because the dev subset also changes with the shard, the ten validation rows are not a conventional fixed-dev learning curve.
9. **Architecture.** Part-specific Spatial GCNs encode body, shared-weight right/left hands with side embeddings, and face. Two graph layers use 48 channels. Part fusion feeds a 6-layer, 256-hidden, 8-head BERT with 1,024-dimensional feed-forward blocks. Total parameters are approximately 5.0 million.
10. **Optimization.** AdamW, learning rate `3e-4`, weight decay `0.01`, batch size 16, gradient accumulation 1, gradient clipping 1.0, 1,000 warmup steps, and a cosine schedule with 25,000 configured total steps. Each shard contributes one stage/epoch.
11. **Seeds.** Model/data seed is 42. Frozen codebooks use deterministic seeds 42–45. The paired shuffle seed is 42017. The Colab runner resets the model seed separately for real and shuffled training.
12. **Checkpoint selection.** `last.pt` is used for shard-to-shard continuation. `best.pt` is selected by the lowest condition-matched validation masked-cluster loss. Since dev shards change, this is a lowest-across-shard-dev criterion, not fixed-dev early stopping.
13. **Lexical extraction.** ASL Citizen skeletons use the portable normalized NPZ format. For each of six layers, valid frame states are mean-pooled. Strict lexical retrieval holds out every token from one signer when constructing concept prototypes.

## Implemented context manipulation

The current four-way frame cluster sequence does not define clean unified spans. Run-length encoding the Cartesian product of four noisy frame labels could collapse to one-frame spans and would violate the requirement to preserve local motion. The implemented, explicit alternative is therefore:

1. Split every cropped sentence into fixed 16-frame local trajectory blocks; the final remainder is retained intact.
2. Never reorder, interpolate, reverse, or drop frames inside a block.
3. In each deterministic source-grouped batch, reassign complete duration-matched blocks to sentence slots.
4. Use every block exactly once, preserving sentence length, total frames, block duration distribution, and all part-cluster marginals exactly within the batch.
5. The assignment strongly forbids donors from the same sentence, prefers the same source video where possible, and includes a boundary-transition matching cost.
6. Mask one frame on each side of every 16-frame join from Transformer attention and masked-cluster supervision in **both** conditions.
7. The `real` replication retains original block order; `shuffled` destroys natural block order and sentence-level co-occurrence.
8. The exact batch IDs, lengths, seeds, and donor assignment indices are saved for reconstruction.

Before full training, each shard must pass transparent integrity gates: exact unigram preservation, at least 90% of blocks moved, at least 90% cross-sentence block assignment, and a shuffled/real first-visible-frame boundary-jump ratio between 0.5 and 2.0. Bigram, mutual-information, and sentence-co-occurrence changes remain reported as descriptive manipulation checks, but are not hard gates: raw count correlations can remain high because the control deliberately preserves strongly non-uniform unit marginals. The hard thresholds only prevent an obviously broken manipulation; they are not scientific significance criteria.

The historical Spatial-SHuBERT remains a reference only. The matched boundary attention mask and source-grouped batching mean that the causal comparison must use the newly trained `real` replication versus the new `shuffled` model.

## Unavoidable limitations

- A fixed 16-frame block is a local visual trajectory, not a validated lexical unit.
- Source-video identity is not guaranteed to equal signer identity.
- Cross-sentence reassignment can create unnatural joins. Matched boundary gaps, source preference, and transition-cost matching reduce but cannot prove elimination of this artefact. The diagnostic therefore reports immediate and first-visible-frame boundary jumps.
- Context is disrupted only among examples that co-occur in a deterministic batch. Source-grouped batching increases same-source reassignment but limits the global donor pool.
- Both paired conditions use a batching/boundary-control pipeline that differs from the historical checkpoint.
- One paired seed is sufficient to run the pipeline but not to support a framework-level scientific conclusion; at least three paired seeds should eventually be run without changing the primary test.

## Colab training

Open the notebook from GitHub after the experiment commit is pushed:

`https://colab.research.google.com/github/ss-sebastian/youtube-asl-skeleton-bert/blob/agent/shape-aware-stgcn/notebooks/colab_context_real_vs_shuffled.ipynb`

Choose a GPU runtime and use **Run all**. It will:

1. install dependencies and run context/Spatial-SHuBERT unit tests;
2. fit one shared skeleton codebook on shard 1;
3. run 64-train/32-validation paired smoke training;
4. generate a full context-manipulation audit for every shard;
5. train real and shuffled once on the same downloaded shard;
6. download one resume bundle containing both conditions after every shard;
7. delete only a shard completed by both conditions.

To resume after a disconnect, upload the latest `context_real_vs_shuffled_after_shard_XX.zip` into `/content`, then Run all again.

## Local evaluation after Colab training

Assume the final bundle is extracted to:

```text
data/checkpoints/context_real_vs_shuffled/
```

### 1. Extract the six lexical layers

```bash
.venv-eeg/bin/python -m sign_semantics.lexical_extract \
  --checkpoint data/checkpoints/context_real_vs_shuffled/spatial_shubert_real_context/checkpoints/best.pt \
  --manifest data/processed/asl_citizen_neural_subset/lexical_evaluation_manifest.csv \
  --root . \
  --output outputs/rsa/context_real_vs_shuffled/real/sign_layers.npz \
  --token-output outputs/rsa/context_real_vs_shuffled/real/sign_layers_tokens.npz

.venv-eeg/bin/python -m sign_semantics.lexical_extract \
  --checkpoint data/checkpoints/context_real_vs_shuffled/spatial_shubert_shuffled_context/checkpoints/best.pt \
  --manifest data/processed/asl_citizen_neural_subset/lexical_evaluation_manifest.csv \
  --root . \
  --output outputs/rsa/context_real_vs_shuffled/shuffled/sign_layers.npz \
  --token-output outputs/rsa/context_real_vs_shuffled/shuffled/sign_layers_tokens.npz
```

### 2. Strict cross-signer lexical comparison

```bash
.venv-eeg/bin/python scripts/run_context_lexical_comparison.py \
  --manifest data/processed/asl_citizen_neural_subset/lexical_evaluation_manifest.csv \
  --real outputs/rsa/context_real_vs_shuffled/real/sign_layers_tokens.npz \
  --shuffled outputs/rsa/context_real_vs_shuffled/shuffled/sign_layers_tokens.npz \
  --random outputs/rsa/objective_comparison/random_spatial_shubert_seed42/sign_layers_tokens.npz \
  --root . \
  --output-dir outputs/lexical/context_real_vs_shuffled \
  --permutations 10000
```

### 3. Aggregate tokens only within exact ASL-LEX codes

```bash
.venv-eeg/bin/python scripts/aggregate_context_exact_codes.py \
  --tokens outputs/rsa/context_real_vs_shuffled/real/sign_layers_tokens.npz \
  --manifest data/processed/asl_citizen_neural_subset/lexical_evaluation_manifest.csv \
  --output data/processed/ds005565/human_associations/context_real_exact_code_layers.npz

.venv-eeg/bin/python scripts/aggregate_context_exact_codes.py \
  --tokens outputs/rsa/context_real_vs_shuffled/shuffled/sign_layers_tokens.npz \
  --manifest data/processed/asl_citizen_neural_subset/lexical_evaluation_manifest.csv \
  --output data/processed/ds005565/human_associations/context_shuffled_exact_code_layers.npz
```

### 4. Primary Deaf-PPMI causal test

```bash
.venv-eeg/bin/python scripts/run_context_semantic_comparison.py \
  --human-rdm data/processed/ds005565/human_associations/human_association_ppmi_rdm.npz \
  --target-name ppmi \
  --real data/processed/ds005565/human_associations/context_real_exact_code_layers.npz \
  --shuffled data/processed/ds005565/human_associations/context_shuffled_exact_code_layers.npz \
  --random data/processed/ds005565/human_associations/random_spatial_shubert_exact_code_layers.npz \
  --raw data/processed/ds005565/human_associations/raw_kinematics_exact_code_layers.npz \
  --output-dir outputs/human_associations/context_real_vs_shuffled \
  --permutations 10000
```

Repeat the same command, changing `--human-rdm` and `--target-name`, for the prespecified robustness targets:

- `human_association_svd_rdm.npz` / `svd`
- `human_association_direct_rdm.npz` / `direct`
- `deaf_iconicity_exact_code_rdm.npz` / `deaf_iconicity`

### 5. Final six-layer table

```bash
.venv-eeg/bin/python scripts/build_context_layer_summary.py \
  --lexical outputs/lexical/context_real_vs_shuffled/lexical_layerwise.csv \
  --semantic outputs/human_associations/context_real_vs_shuffled/semantic_ppmi_real_minus_shuffled_tests.csv \
  --output outputs/context_real_vs_shuffled_layer_summary.csv
```

### 6. Matched learning curves

```bash
.venv-eeg/bin/python scripts/summarize_context_training.py \
  --real data/checkpoints/context_real_vs_shuffled/spatial_shubert_real_context/checkpoints/metrics.csv \
  --shuffled data/checkpoints/context_real_vs_shuffled/spatial_shubert_shuffled_context/checkpoints/metrics.csv \
  --output outputs/context_manipulation/matched_training_metrics.csv
```

## Primary inference rule

The prespecified semantic statistic is the one-sided, shared-permutation difference

\[
\rho_{real,\ Deaf\ PPMI} - \rho_{shuffled,\ Deaf\ PPMI},
\]

with max-statistic FWER correction across all six layers. A significant raw RSA for the real model is not a context effect unless this paired difference is also significant in the predicted direction. SVD, direct association, and Deaf iconicity remain robustness analyses and do not alter the primary PPMI family.
