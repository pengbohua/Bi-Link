#!/usr/bin/env bash

set -x
set -e

DATA_DIR="data/wiki5m_ind"
model_path="checkpoints/wiki5m_ind/model_best.mdl"


python3 -u evaluate.py \
--task "wiki5m_ind" \
--is-test \
--eval-model-path "${model_path}" \
--rel-path "${DATA_DIR}/relations.json" \
--prefix-seq-len 50 \
--rerank-n-hop 5 \
--num-rels 822 \
--neighbor-weight 0.05 \
--train-path "$DATA_DIR/train.txt.json" \
--valid-path "$DATA_DIR/test.txt.json" "$@"
