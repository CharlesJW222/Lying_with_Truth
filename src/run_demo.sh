#!/usr/bin/env bash

# victims to test
VICTIMS=(
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
  "Qwen/Qwen2.5-14B-Instruct"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
  "Qwen/Qwen2.5-3B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct"
  "claude-3-5-haiku-20241022"
  "claude-3-haiku-20240307"
  "claude-haiku-4-5-20251001"
)

for victim in "${VICTIMS[@]}"; do
  echo "============================================"
  echo "Running victim: $victim"
  echo "============================================"

  python demo.py --event charliehebdo --victim_name "$victim"

  echo "Finished victim: $victim"
  echo
done
