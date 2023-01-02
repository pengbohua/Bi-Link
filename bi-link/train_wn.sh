#!/usr/bin/env bash

set -x
set -e

TASK="WN18RR"
DATA_DIR="../SimKGC-main/data"
OUTPUT_DIR="../checkpoint/bilink_wn18rr"
CUDA_VISIBLE_DEVICES="0,1"
DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/checkpoint/${TASK}_$(date +%F-%H%M.%S)"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/${TASK}"
fi

python3 -u main.py \
--model-dir "${OUTPUT_DIR}" \
--pretrained-model ../pretrained_models/bert-base \
--pooling mean \
--lr 5e-5 \
--t 20 \
--hr-tr-s 15 \
--use-link-graph \
--train-path "${DATA_DIR}/${TASK}/train.txt.json" \
--valid-path "${DATA_DIR}/${TASK}/valid.txt.json" \
--test-path "${DATA_DIR}/${TASK}/test.txt.json" \
--task ${TASK} \
--batch-size 512 \
--print-freq 20 \
--additive-margin 0.02 \
--use-amp False \
--epochs 12 \
--workers 4 \
--max-to-keep 3 "$@"

OUTPUT_DIR="../checkpoint/bilink_wn18rr"
python3 -u main.py \
	--model-dir "${OUTPUT_DIR}" \
	--pretrained-model ../pretrained_models/bert-base \
	--pooling mean \
	--lr 5e-5 \
	--t 20 \
	--hr-tr-s 15 \
	--use-link-graph \
	--train-path "${DATA_DIR}/${TASK}/train.txt.json" \
	--valid-path "${DATA_DIR}/${TASK}/valid.txt.json" \
	--test-path "${DATA_DIR}/${TASK}/test.txt.json" \
	--task ${TASK} \
	--batch-size 512 \
	--print-freq 20 \
	--additive-margin 0.02 \
	--use-amp False \
	--epochs 12 \
	--workers 4 \
	--off-diag True \
	--max-to-keep 3 "$@"
