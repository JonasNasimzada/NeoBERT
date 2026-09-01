# FineWeb-Edu paired Long Range Arena evaluation

This evaluation compares the exact 99,985,152-parameter FineWeb-Edu
multispace-Flash and real-Flash checkpoints. It is an **external paired
pretrained adaptation**, not an official LRA leaderboard submission. The
official apples-to-apples setting trains the much smaller task-specific LRA
architectures with fixed model dimensions; these pretrained 100M models do not
meet that setting.

## Capacity and task coverage

The checkpoints use RoPE and were trained with `max_length=1024`. No task is
silently truncated.

| Canonical task | Length | Status |
|---|---:|---|
| ListOps | 2,000 | Unsupported: exceeds native context |
| Byte-level IMDB Text | 4,000 | Unsupported: exceeds native context |
| AAN Retrieval | 4,000 per document | Unsupported: exceeds native context; underlying document text is not redistributed by LRA |
| Sequential CIFAR-10 Image | 1,024 pixels + CLS | Running as an explicitly external one-position RoPE adaptation |
| Pathfinder-32 | 1,024 | Capacity-compatible, but not run because the official data release is currently inaccessible |
| Path-X | 16,384 | Unsupported: exceeds native context |

The official repository's legacy Text config contains a 1,000-byte default,
while the canonical paper benchmark is the 4K byte-level task. This evaluation
does not substitute the shorter legacy default and call it the canonical Text
score.

The official LRA release URL
`https://storage.googleapis.com/long-range-arena/lra_release.gz` returned HTTP
403 when this suite was prepared on 2026-08-31. A community Pathfinder mirror
was inspected but rejected because it had no provenance and exposed anomalous
199,999/200,000-row arrays. Pathfinder remains `data_unavailable` rather than
using an unverifiable substitute.

The machine-readable coverage declaration is
[`conf/evaluation/lra_fineweb_external.json`](conf/evaluation/lra_fineweb_external.json).
No six-task average is reported for this incomplete suite.

## Runnable Image protocol

The official LRA Image pipeline converts CIFAR-10 to grayscale, flattens the
32x32 image into 1,024 row-major pixels, prepends a classifier token, and uses
CLS pooling. Preserving all pixels therefore requires 1,025 internal positions.
Because RoPE frequencies are analytical, nonpersistent buffers, the adapter
regenerates frequencies at length 1,025 without adding or resizing any learned
checkpoint tensor. This identical one-position extension is applied to both
models and is recorded in every result.

The pretrained BERT CLS vocabulary ID is prepended. Pixel values are mapped
injectively as `token_id = uint8_pixel + 1000`, avoiding the checkpoint's PAD
ID. There is no padding, masking, augmentation, or truncation. The complete
99,985,152-parameter backbone and an identically initialized Dense-ReLU-Dense
classifier head are fine-tuned.

Data source:

- Hub dataset: `uoft-cs/cifar10`
- revision: `0b2714987fa478483af9968de7c934580d0bb9a2`
- train parquet SHA256: `8428b53a88a11ac374111006708df51469e315a22ac6d66470afd9c78d2ae883`
- test parquet SHA256: `841389e6f2d64f28bf17310e430aebac20ec3ba611a3c5e231dc93c645ce84de`
- split: first 45,000 source-training rows for training, final 5,000 for validation, and all 10,000 test rows for testing

The deterministic preprocessing manifest is at
`/mnt/nfs/home/st171793/ComplexAttention/.cache/evaluations/lra/cifar10-prepared/manifest.json`.

Full hyperparameters are fixed across variants and seeds:

- seeds 42, 43, and 44
- five epochs; no early stopping of the schedule
- microbatch 4, gradient accumulation 4, nominal effective batch 16
- AdamW beta1=0.9, beta2=0.98, epsilon=1e-9, weight decay 0
- backbone learning rate 3e-5; classifier-head learning rate 5e-4
- one-epoch linear warmup followed by cosine decay
- BF16 autocast, FP32 cross-entropy, gradient norm clipping at 1.0
- native FlashAttention implementation on an NVIDIA A100
- best validation epoch selected once, then evaluated on the untouched test set

For each seed, the two variants use identical classifier initialization,
shuffled example order, optimizer settings, and data fingerprints. The final
report gives means, sample standard deviations, and paired multispace-minus-real
differences with a 95% Student-t interval over the three seeds.

## Files and jobs

- Data preparation: `scripts/fineweb_evaluation/lra_prepare.py`
- Training/evaluation: `scripts/fineweb_evaluation/lra.py`
- Aggregation: `scripts/fineweb_evaluation/lra_aggregate.py`
- A100 preflight: `jobs/fineweb_evaluation/lra_preflight.sbatch`
- Full paired array: `jobs/fineweb_evaluation/lra.sbatch`
- Aggregate job: `jobs/fineweb_evaluation/lra_aggregate.sbatch`
- Gated submission: `jobs/fineweb_evaluation/submit_lra.sh`
- Results: `logs/fineweb_evaluation/lra_external/`

Submitted chain: preflight `67623`, full array `67624`, aggregate `67625`.
