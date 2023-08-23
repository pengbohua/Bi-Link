#!/usr/bin/env bash

set -x
set -e

DATA_DIR="PathToRepo/data/WN18RR"
model_path="PathToRepo/checkpoints/wn18rr/model_best.mdl"
RELATION_PATH="${DATA_DIR}/relations.json"


python3 -u evaluate.py \
--task "WN18RR" \
--is-test \
--eval-model-path "${model_path}" \
--rel-path "${RELATION_PATH}" \
--prefix-seq-len 80 \
--rerank-n-hop 5 \
--num-rels 822 \
--neighbor-weight 0.05 \
--train-path "$DATA_DIR/train.txt.json" \
--valid-path "$DATA_DIR/test.txt.json" "$@"
