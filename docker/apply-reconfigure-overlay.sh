#!/usr/bin/env bash
# Runtime patch: copy reconfigure API files from a mounted vLLM checkout onto
# the installed vllm package. Run this before starting vllm serve.
#
# Usage:
#   docker run ... -v /path/to/vllm:/workspace/vllm-patch:ro ...
#   bash /workspace/vllm-patch/docker/apply-reconfigure-overlay.sh
set -euo pipefail

PATCH_DIR="${1:-/workspace/vllm-patch}"
VLLM_DIR=$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))")

FILES=(
    "v1/core/sched/interface.py"
    "v1/core/sched/scheduler.py"
    "v1/engine/__init__.py"
    "v1/engine/core.py"
    "v1/engine/core_client.py"
    "v1/engine/async_llm.py"
    "engine/protocol.py"
    "entrypoints/serve/rlhf/api_router.py"
)

echo "Patching $VLLM_DIR from $PATCH_DIR/vllm/"
for f in "${FILES[@]}"; do
    src="${PATCH_DIR}/vllm/${f}"
    dst="${VLLM_DIR}/${f}"
    if [[ -f "$src" ]]; then
        cp "$src" "$dst"
        echo "  patched $f"
    else
        echo "  WARN: $src not found, skipping"
    fi
done
echo "Reconfigure API overlay applied."
