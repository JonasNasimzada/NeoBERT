#!/usr/bin/env bash
#SBATCH --job-name=ca-s1024-test
#SBATCH --partition=slowlane
#SBATCH --gpus=A100:1
#SBATCH --qos=hiwi_project
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=ca-s1024-test-%j.out
#SBATCH --error=ca-s1024-test-%j.err

set -euo pipefail

if [[ -n "${NEOBERT_ROOT:-}" ]]; then
    job_dir="$NEOBERT_ROOT/jobs/attention_ablation"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    job_dir="$SLURM_SUBMIT_DIR/jobs/attention_ablation"
else
    job_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
source "$job_dir/common.sh"
export ATTENTION_SPACE=multispace
export ATTENTION_BACKEND=flash
setup_attention_runtime
export PYTHONDONTWRITEBYTECODE=1

cd "$COMPLEX_ATTENTION_ROOT"
bash -n \
    NeoBERT/jobs/attention_ablation/common.sh \
    NeoBERT/jobs/attention_ablation/prepare_data.sbatch \
    NeoBERT/jobs/attention_ablation/train.sbatch \
    NeoBERT/jobs/attention_ablation/benchmark.sbatch \
    NeoBERT/jobs/attention_ablation/benchmark_attention_papers.sbatch \
    NeoBERT/jobs/attention_ablation/submit.sh \
    NeoBERT/jobs/real_fineweb/train.sbatch \
    NeoBERT/jobs/real_fineweb/submit.sh

"$COMPLEX_ATTN_PYTHON" \
    NeoBERT/scripts/attention_ablation/validate_variants.py

"$COMPLEX_ATTN_PYTHON" \
    NeoBERT/scripts/attention_ablation/validate_100m_pair.py

"$COMPLEX_ATTN_PYTHON" \
    NeoBERT/scripts/attention_ablation/check_runtime.py \
    --space multispace --backend flash --require-a100

"$COMPLEX_ATTN_PYTHON" \
    NeoBERT/scripts/attention_ablation/check_runtime.py \
    --space real --backend flash --require-a100

"$COMPLEX_ATTN_PYTHON" -m pytest -q -p no:cacheprovider \
    test/test_dual_flash_attention.py \
    NeoBERT/tests/test_attention_ablation_variants.py \
    NeoBERT/tests/test_attention_ablation_benchmarks.py \
    NeoBERT/tests/test_attention_ablation_kernel_benchmark.py \
    NeoBERT/tests/test_multispace_fineweb_training.py \
    NeoBERT/tests/test_real_fineweb_training.py \
    NeoBERT/tests/test_optibertneo_data_packing.py \
    NeoBERT/tests/test_pretraining_observability.py \
    NeoBERT/tests/test_tokenizer_configuration.py

"$COMPLEX_ATTN_PYTHON" -m pytest -q -p no:cacheprovider \
    NeoBERT/tests/test_complex_attention_integration.py::TestComplexAttentionIntegration::test_config_enforces_backend_matrix \
    NeoBERT/tests/test_complex_attention_integration.py::TestComplexAttentionIntegration::test_config_rejects_nonpositive_attention_geometry \
    NeoBERT/tests/test_complex_attention_integration.py::TestComplexAttentionIntegration::test_config_validates_attention_dropout \
    NeoBERT/tests/test_complex_attention_integration.py::TestComplexAttentionIntegration::test_multispace_attention_partitions_one_packed_qkv_and_concatenates \
    NeoBERT/tests/test_complex_attention_integration.py::TestComplexAttentionIntegration::test_complex_flash_backends_take_low_precision_optimizer_steps \
    NeoBERT/tests/test_complex_attention_integration.py::TestComplexAttentionIntegration::test_multispace_flash_model_takes_low_precision_optimizer_step \
    NeoBERT/tests/test_complex_attention_integration.py::TestComplexAttentionIntegration::test_multispace_flash_model_runs_production_geometry
