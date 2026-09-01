#!/usr/bin/env bash
#SBATCH --job-name=optibertneo-pair-test
#SBATCH --partition=slowlane
#SBATCH --gpus=A100:1
#SBATCH --qos=hiwi_project
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=optibertneo-pair-test-%j.out
#SBATCH --error=optibertneo-pair-test-%j.err

set -euo pipefail

# TEST ONLY: this allocation never preprocesses FineWeb-Edu, invokes the
# pretraining entry point, writes a checkpoint, or submits another Slurm job.
if [[ -n "${NEOBERT_ROOT:-}" ]]; then
    job_dir="$NEOBERT_ROOT/jobs/slurm"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    job_dir="$SLURM_SUBMIT_DIR/jobs/slurm"
else
    job_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
common_file="$job_dir/../attention_ablation/common.sh"
if [[ ! -f "$common_file" ]]; then
    echo "Cannot locate jobs/attention_ablation/common.sh from $job_dir" >&2
    echo "Run from the NeoBERT root or export NEOBERT_ROOT explicitly." >&2
    exit 2
fi

source "$common_file"
export ATTENTION_SPACE=multispace
export ATTENTION_BACKEND=flex
setup_attention_runtime
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "$NEOBERT_ROOT"
git diff --check -- \
    README.md \
    OPTIBERTNEO.md \
    conf/dataset/fineweb_edu.yaml \
    conf/model/optibertneo-198m.yaml \
    jobs/optibertneo-1p3b.sh \
    scripts/pretraining/inspect_optibertneo.py \
    scripts/pretraining/preflight_optibertneo.py \
    scripts/pretraining/preprocess.py \
    src/neobert/model/complex_attention.py \
    src/neobert/model/model.py \
    src/neobert/pretraining/trainer.py \
    tests/test_complex_attention_integration.py \
    tests/test_multispace_cuda_streams.py \
    tests/test_optibertneo_data_packing.py \
    tests/test_optibertneo_training_setup.py \
    tests/test_pretraining_observability.py

# Shell parsing and meta/static checks still run inside this A100 allocation so
# the complete validation workflow honors the GPU-only test policy.
bash -n \
    jobs/optibertneo-1p3b.sh \
    jobs/submit-optibertneo-pair.sh \
    jobs/slurm/optibertneo-paired-1p3b-2n8g.sbatch \
    jobs/slurm/run-optibertneo-paired-node.sh \
    jobs/slurm/test-optibertneo-pair-a100.sbatch

"$COMPLEX_ATTN_PYTHON" scripts/pretraining/validate_optibertneo_pair.py
"$COMPLEX_ATTN_PYTHON" scripts/pretraining/preflight_optibertneo.py \
    --config-only --variant real
"$COMPLEX_ATTN_PYTHON" scripts/pretraining/preflight_optibertneo.py \
    --config-only --variant multispace

"$COMPLEX_ATTN_PYTHON" -m pytest -q -p no:cacheprovider \
    tests/test_optibertneo_pair.py \
    tests/test_optibertneo_checkpointing.py \
    tests/test_optibertneo_training_setup.py \
    tests/test_optibertneo_data_packing.py \
    tests/test_pretraining_observability.py \
    tests/test_multispace_cuda_streams.py \
    tests/test_complex_attention_integration.py::TestComplexAttentionIntegration::test_config_enforces_backend_matrix \
    tests/test_complex_attention_integration.py::TestComplexAttentionIntegration::test_multispace_flex_routes_one_document_block_mask_to_all_spaces

run_smoke() {
    local variant="${1:?variant is required}"
    local layers="${2:?layer count is required}"
    local sequence_length="${3:?sequence length is required}"
    local compile_mode="${4:-0}"
    local batch_size="${5:-1}"
    local compile_args=()
    local device_args=()
    if [[ "$compile_mode" == 1 ]]; then
        compile_args=(--compile)
    fi
    case "${REQUIRE_A100:-1}" in
        1) device_args=(--require-a100) ;;
        0) ;;
        *)
            echo "REQUIRE_A100 must be 0 or 1" >&2
            return 2
            ;;
    esac
    echo "GPU smoke: variant=$variant layers=$layers sequence_length=$sequence_length batch_size=$batch_size compile=$compile_mode"
    "$COMPLEX_ATTN_PYTHON" scripts/pretraining/smoke_optibertneo_pair.py \
        --variant "$variant" \
        --layers "$layers" \
        --sequence-length "$sequence_length" \
        --batch-size "$batch_size" \
        "${device_args[@]}" \
        "${compile_args[@]}"
}

# Long-context kernel coverage with one production-width block.
run_smoke real 1 1024 1
run_smoke multispace 1 1024 1

# Full 28-layer graph coverage at a conservative context length. The optional
# production-geometry gate below additionally covers the dense dual tangent/JVP
# path at the launcher's default multispace microbatch.
run_smoke real 28 64
run_smoke multispace 28 64

case "${FULL_PRODUCTION_GEOMETRY:-0}" in
    0) ;;
    1)
        # Optional decisive production-geometry check. It remains one
        # synthetic optimizer step per variant and never enters training.
        run_smoke real 28 1024 1
        run_smoke multispace 28 1024 1 "${MULTISPACE_PRODUCTION_BATCH:-8}"
        ;;
    *)
        echo "FULL_PRODUCTION_GEOMETRY must be 0 or 1" >&2
        exit 2
        ;;
esac

echo "OptiBERTneo paired GPU validation passed; no data or training job was submitted."
