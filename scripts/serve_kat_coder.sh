#!/usr/bin/env bash
# Launch local Kat-Coder v2.5 on AMD Strix Point iGPU with Vulkan acceleration

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/olivermikkola/Models/Kwaipilot_KAT-Coder-V2.5-Dev-Q4_K_M.gguf}"
LLAMA_BUILD="${LLAMA_BUILD:-/home/olivermikkola/Projects/llama.cpp/build/bin}"
PORT="${PORT:-8080}"
CTX="${CTX:-16384}"
NGL="${NGL:-99}"

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "Error: model file not found at $MODEL_PATH" >&2
    exit 1
fi

if [[ ! -f "$LLAMA_BUILD/llama-server" ]]; then
    echo "Error: llama-server not found at $LLAMA_BUILD/llama-server" >&2
    exit 1
fi

export LD_LIBRARY_PATH="$LLAMA_BUILD:${LD_LIBRARY_PATH:-}"

echo "=========================================================="
echo " Starting Kat-Coder v2.5 Dev (36B MoE Q4_K_M)"
echo " Acceleration: AMD Strix Point [Radeon 880M/890M] via Vulkan"
echo " Model: $MODEL_PATH"
echo " Server: http://127.0.0.1:$PORT"
echo " Context: $CTX tokens"
echo " Offload: $NGL layers to iGPU"
echo "=========================================================="
echo ""
echo "To connect Keel to this server in another terminal, run:"
echo "  export KEEL_BASE_URL=\"http://127.0.0.1:$PORT\""
echo "  export KEEL_MODEL=\"kat-coder\""
echo "  keel repl"
echo "=========================================================="
echo ""

exec "$LLAMA_BUILD/llama-server" \
    -m "$MODEL_PATH" \
    --host "127.0.0.1" \
    --port "$PORT" \
    -c "$CTX" \
    -ngl "$NGL" \
    --alias "kat-coder" \
    --parallel 1
