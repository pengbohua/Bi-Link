#!/usr/bin/env bash

set -x
set -e

TASK="wiki5m_ind"
if [[ $# -ge 1 ]]; then
    TASK=$1
    shift
fi

python3 -u preprocess.py \
--task "${TASK}" \
--train-path "data/wiki5m_ind/train.txt" \
--valid-path "data/wiki5m_ind/valid.txt" \
--test-path "data/wiki5m_ind/test.txt"
