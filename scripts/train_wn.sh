#!/usr/bin/env bash

set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/checkpoint/${TASK}_$(date +%F-%H%M.%S)"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/${TASK}"
fi

python main.py --model-dir "${OUTPUT_DIR}" \
--pretrained-model bert-base-uncased \
--pooling mean \
--lr 1e-3 \
--use-link-graph \
--train-path "${DATA_DIR}/train.txt.json" \
--valid-path "${DATA_DIR}/valid.txt.json" \
--rel-path "${DATA_DIR}/relations.json" \
--num-rels 11 \
--task WN18RR \
--batch-size 1024 \
--print-freq 20 \
--additive-margin 0.02 \
--use-amp \
--use-self-negative \
--finetune-t \
--epochs 1 \
--workers 4 \
--max-to-keep 5 \
--prefix-seq-len 80