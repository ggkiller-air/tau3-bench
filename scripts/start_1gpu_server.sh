#!/bin/bash
# 1卡 vLLM 服务器启动脚本
# 目标：另起一个新 server 做实验，不 kill 之前的 server，并避免端口冲突

set -euo pipefail

echo "=========================================="
echo "1-GPU Tensor Parallel vLLM Server"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
echo "Using GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "=========================================="

# Keep in sync with other local-model scripts
model_list=(
  "qwen3-8b"
  "qwen3-4b-instruct-2507"
  "qwen3-4b-instruct-2507-fp8"
  "qwen3-4b-instruct-2507-gptq-int4"
  "qwen3-1.7b"
  "qwen3-0.6b"
  "deepseek-r1-distill-qwen-1.5b"
  "llama-3.2-1b-instruct"
  "llama-3.2-3b-instruct"
  "qwen3-instruct-8b-custom"
)

# Defaults (can be overridden by args/env)
MODEL_NAME="${MODEL_NAME:-deepseek-r1-distill-qwen-1.5b}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}" # 1-GPU 默认使用 8000；如需多开实例请通过 --port (或 PORT=...) 指定其他端口
TP_SIZE="${TP_SIZE:-1}"
# 默认对外暴露的模型名：用原始 MODEL_NAME，避免你请求 `qwen3-4b-instruct-2507` 时 404
# 如需区分多个实例，可用 --served-model-name 显式指定
SERVED_MODEL_NAME=""  # 解析推迟到参数解析之后（见下），跟随最终 MODEL_NAME；不继承环境变量，避免残留名污染

extra_vllm_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --list-models)
      printf "%s\n" "${model_list[@]}"
      exit 0
      ;;
    --model|--model-name)
      MODEL_NAME="${2:?Missing value for $1}"
      shift 2
      ;;
    --served-model-name)
      SERVED_MODEL_NAME="${2:?Missing value for $1}"
      shift 2
      ;;
    --max-model-len)
      MAX_MODEL_LEN="${2:?Missing value for $1}"
      shift 2
      ;;
    --host)
      HOST="${2:?Missing value for $1}"
      shift 2
      ;;
    --port)
      PORT="${2:?Missing value for $1}"
      shift 2
      ;;
    --tensor-parallel-size|--tp|--tp-size)
      TP_SIZE="${2:?Missing value for $1}"
      shift 2
      ;;
    *)
      extra_vllm_args+=("$1")
      shift
      ;;
  esac
done

# 对外服务名跟随实际 MODEL_NAME；仅当显式传 --served-model-name 时才覆盖。
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL_NAME}}"

if [[ "${TP_SIZE}" != "1" ]]; then
  echo "⚠️  当前脚本默认用于 1 卡（TP_SIZE=1）。你设置了 TP_SIZE=${TP_SIZE}，请确保与 CUDA_VISIBLE_DEVICES 的卡数一致。"
fi

echo ""
echo "Checking port availability..."
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:|\\[)${PORT}\$"; then
  echo "❌ Port ${PORT} is already in use. Please choose another port via --port (or PORT=...)."
  exit 1
fi

echo ""
echo "Checking GPU availability..."
GPU_COUNT="$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | head -n "${GPU_COUNT}" || true

# Model path mapping
case "$MODEL_NAME" in
  qwen3-8b)
    MODEL_PATH="/mnt/ssd2/jhn/ECAgent-project/models/Qwen3-8B/Qwen/Qwen3-8B"
    ;;
  qwen3-4b-instruct-2507)
    MODEL_PATH="/mnt/ssd2/jhn/ECAgent-project/models/Qwen3-4B-Instruct-2507/Qwen/Qwen3-4B-Instruct-2507"
    ;;
  qwen3-4b-instruct-2507-fp8)
    MODEL_PATH="/mnt/ssd2/jhn/ECAgent-project/models/Qwen3-4B-Instruct-2507-FP8/Qwen/Qwen3-4B-Instruct-2507-FP8"
    ;;
  qwen3-4b-instruct-2507-gptq-int4)
    MODEL_PATH="/mnt/ssd2/jhn/ECAgent-project/models/Qwen3-4B-Instruct-2507-GPTQ-Int4/JunHowie/Qwen3-4B-Instruct-2507-GPTQ-Int4"
    ;;
  qwen3-1.7b)
    MODEL_PATH="/mnt/ssd2/jhn/ECAgent-project/models/Qwen3-1.7B/Qwen/Qwen3-1.7B"
    ;;
  qwen3-0.6b)
    MODEL_PATH="/mnt/ssd2/jhn/ECAgent-project/models/Qwen3-0.6B/Qwen/Qwen3-0.6B"
    ;;
  deepseek-r1-distill-qwen-1.5b)
    MODEL_PATH="/mnt/ssd2/jhn/ECAgent-project/models/DeepSeek-R1-Distill-Qwen-1.5B/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    ;;
  llama-3.2-1b-instruct)
    MODEL_PATH="/mnt/ssd2/jhn/ECAgent-project/models/Llama-3.2-1B-Instruct/meta-llama/Llama-3.2-1B-Instruct"
    ;;
  llama-3.2-3b-instruct)
    MODEL_PATH="/mnt/ssd2/jhn/ECAgent-project/models/Llama-3.2-3B-Instruct/meta-llama/Llama-3.2-3B-Instruct"
    ;;
  qwen3-instruct-8b-custom)
    MODEL_PATH="/mnt/ssd0/qwen-8b-instruct"
    ;;
  *)
    echo "❌ Unknown model: $MODEL_NAME"
    echo "Available models:"
    printf "  - %s\n" "${model_list[@]}"
    exit 1
    ;;
esac

LOG_FILE="vllm_server_${MODEL_NAME}_${TP_SIZE}gpu_port${PORT}.log"

echo ""
echo "Starting ${TP_SIZE}-GPU Tensor Parallel server..."
echo "Host: ${HOST}"
echo "Port: ${PORT}"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "This may take 2-3 minutes to initialize..."
echo ""

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --tensor-parallel-size "${TP_SIZE}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 512 \
    --gpu-memory-utilization 0.9 \
    --enable-chunked-prefill \
    "${extra_vllm_args[@]}" \
    > "${LOG_FILE}" 2>&1 &

SERVER_PID=$!
echo "Server started with PID: $SERVER_PID"
echo "📝 Log file: ${LOG_FILE}"
echo ""

echo "Waiting for server to initialize (this may take 2-3 minutes)..."
echo "You can monitor progress with: tail -f ${LOG_FILE}"
echo ""

for i in {1..18}; do
    sleep 10
    if curl -s "http://${HOST}:${PORT}/v1/models" > /dev/null 2>&1; then
        echo ""
        echo "✅ Server is ready!"
        break
    fi
    echo -n "."
done

echo ""
echo "=========================================="
echo "Server Status:"
echo "=========================================="

if curl -s "http://${HOST}:${PORT}/v1/models" > /dev/null 2>&1; then
    echo "✅ API is responding"
    curl -s "http://${HOST}:${PORT}/v1/models" | python -m json.tool | grep -E "(id|max_model_len|tensor_parallel)" || true
    echo ""
    echo "📊 GPU Usage:"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader | head -n "${GPU_COUNT}" || true
    echo ""
    echo "🧪 Test: curl -s http://${HOST}:${PORT}/v1/models | python -m json.tool"
else
    echo "⚠️  Server may still be initializing..."
    echo "Check logs: tail -f ${LOG_FILE}"
fi

echo ""
echo "=========================================="
echo "To stop this server: kill ${SERVER_PID}"
echo "=========================================="


