#!/usr/bin/env bash

set -x
set -e

DATA_DIR="data/WN18RR"
model_path="checkpoints/wiki5m_ind/model_best.mdl"


python3 -u evaluate.py \
--task "WN18RR" \
--is-test \
--eval-model-path "${model_path}" \
--rel-path "${DATA_DIR}/relations.json" \
--num-rels 11 \
--rerank-n-hop 5 \
--neighbor-weight 0.05 \
--train-path "$DATA_DIR/train.txt.json" \
--valid-path "$DATA_DIR/test.txt.json" "$@"
