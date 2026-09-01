# GLUE evaluation for the paired FineWeb-Edu models

This evaluation compares the exact 99,985,152-parameter multispace and
real-space checkpoints on every scored GLUE task: CoLA, SST-2, MRPC, STS-B,
QQP, MNLI, QNLI, RTE, and WNLI. MNLI reports matched and mismatched validation
accuracy separately. The GLUE AX diagnostic is also run from every MNLI model,
but its labels are private, so predictions are exported without a score and AX
is excluded from the nine-task aggregate.

These are **public-validation diagnostic results**, not official GLUE
test-server or leaderboard scores. The checkpoints were pretrained on
FineWeb-Edu, and no claim of GLUE training-data exclusion is made.

## Fixed paired protocol

- Dataset: `nyu-mll/glue` revision
  `bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c`. Every parquet is checked against
  its pinned Git-LFS SHA-256 and byte count.
- Seeds: 42, 43, and 44 for both architecture variants.
- Training: three fixed epochs, AdamW, learning rate `2e-5`, weight decay
  `0.01`, six-percent linear warmup, effective batch size 32, gradient clipping
  at 1.0, maximum length 128, and BF16 autocast with FP32 parameters.
- No hyperparameter search and no best-epoch selection: the fixed final epoch
  is scored, preventing validation-set selection from favoring either model.
- The paired head initialization SHA-256 and complete per-epoch training-order
  SHA-256 must match between variants for every task and seed. Aggregation
  fails if any pairing invariant differs.
- The classifier/regressor is a standard BERT-style first-token (`[CLS]`)
  head. Both variants receive the same initialization and dropout stream.
- Checkpoints retain their learned tensors. Variable-length padded batches use
  the mathematically equivalent NeoBERT PyTorch attention backend because the
  strict Flash kernel is padding-free. Both variants use this same supervised
  backend.
- Every forward, backward, and evaluation pass is guarded to require an NVIDIA
  A100. Dataset download/checksumming and result aggregation do not execute a
  model.

## Metrics and aggregate

The local metric implementation matches the pinned Hugging Face Evaluate GLUE
metric at revision `e1a5d749a1772a37a8b68348d29f314a000d7907`:

- CoLA: Matthews correlation.
- SST-2, MNLI matched/mismatched, QNLI, RTE, WNLI: accuracy.
- MRPC and QQP: accuracy and F1; the task score is their mean.
- STS-B: Pearson and Spearman correlation; the task score is their mean.

MNLI's task score is the mean of matched and mismatched accuracy. The overall
score is the unweighted mean of the nine task scores. Reports include means,
sample standard deviations, paired seed deltas, and 95% paired t intervals.

## Jobs and outputs

Run `jobs/fineweb_evaluation/submit_glue.sh`. It downloads/verifies the pinned
snapshot, then submits an A100 paired preflight, a 54-member array
(`2 models x 3 seeds x 9 tasks`), and a dependency-gated aggregate job.

The default output root is
`logs/fineweb_evaluation/glue-validation-v1`. The final files are
`paired_comparison.json` and `summary.md`; individual reports and compressed
prediction/logit arrays live under `<variant>/seed-<seed>/<task>/`.
