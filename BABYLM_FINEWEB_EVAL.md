# FineWeb-Edu Pair: BabyLM 2026 Strict External Evaluation

This workflow runs the official BabyLM 2026 Strict evaluation on the exact
99,985,152-parameter multispace and real-MHA checkpoints. It is an external
diagnostic comparison, **not** a leaderboard-eligible BabyLM submission: the
models were pretrained on 1,376,256,000 FineWeb-Edu token positions and do not
meet the Strict track's 100M-word budget.

## Pinned evaluator and data

- Evaluator: `babylm-org/babylm-eval` at
  `6f825c291e2c4c78ad33b1935fd64d45f52642dc`
- Local evaluator: `/mnt/nfs/home/st171793/babylm-eval-2026`
- Core evaluation data: `BabyLM-community/BabyLM-2026-Strict-Evals` at
  `8d52da9424a9ff30b9e8266c4f751aba9c504233`
- EWoK: `ewok-core/ewok-core-1.0` at
  `34d912a608066c92e2990a0328ffc3bd9a716042`; gated access is already accepted
- GlobalPIQA parallel/nonparallel:
  `b0b18516a8bc2cb1106bce3dd4db32848ca715ea` and
  `6777742fa3634c0583cda3b7f8a482ea7b1b0937`

The machine-readable version is
`conf/evaluation/babylm_2026_strict_external.yaml`.

## Active jobs

- `67507`: paired A100 zero-shot preflight; completed successfully
- `67508`: paired full zero-shot evaluation, array `0=multispace, 1=real`
- `67518`: paired A100 fine-tuning forward/backward preflight
- `67526`: full fine-tune array, dependent on both `67508` and `67518`

The zero-shot battery runs BLiMP, the BLiMP supplement, EWoK, Entity Tracking,
COMPS, GlobalPIQA parallel and nonparallel, and Reading. Results are written to:

```text
logs/fineweb_eval/babylm-2026-strict-external/results/
├── fineweb-edu-multispace-100m-seed-42/
└── fineweb-edu-real-100m-seed-42/
```

## Fine-tuning matrix

Job `67526` runs the official shared-hyperparameter (Super)GLUE recipe:

| Array task | Model | Task |
|---:|---|---|
| 0–6 | multispace | BoolQ, MultiRC, RTE, WSC, MRPC, QQP, MNLI |
| 7–13 | real | BoolQ, MultiRC, RTE, WSC, MRPC, QQP, MNLI |

The model/task order within each seven-task group is exactly
`boolq,multirc,rte,wsc,mrpc,qqp,mnli`.

The checkpoints retain the learned weights but select the PyTorch attention
implementation for fine-tuning. The official batches are padded and FP32,
whereas the direct Flash implementation is deliberately padding-free and
BF16-only. This is an implementation-backend switch, not an architecture or
parameter change, and is recorded in each checkpoint manifest.

Fine-tuned state dictionaries and manifests are saved under:

```text
logs/fineweb_eval/babylm-2026-strict-external/finetuned_models/
├── fineweb-edu-multispace-100m-seed-42/{task}/
│   ├── model.pt
│   └── checkpoint_manifest.json
└── fineweb-edu-real-100m-seed-42/{task}/
    ├── model.pt
    └── checkpoint_manifest.json
```

The `model.pt` files are state dictionaries for the official
`evaluation_pipeline.finetune.classifier_model.ModelForSequenceClassification`
wrapper. The manifests contain the exact base checkpoint, wrapper class,
hyperparameters, pooling rule, backend provenance, and label mapping. In
particular, MNLI uses `0=entailment, 1=neutral, 2=contradiction`, while QQP uses
`0=not_duplicate, 1=duplicate`. These artifacts are preserved for paired
MNLI→HANS and QQP→PAWS transfer evaluation.
