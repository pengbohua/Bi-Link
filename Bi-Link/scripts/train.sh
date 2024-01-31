#!/usr/bin/env bash

set -x
set -e

CUDA_VISIBLE_DEVICES=0,1 python main.py --model-dir checkpoints/wiki5m_ind \
--pretrained-model bert-base-uncased \
--pooling mean \
--lr 1e-3 \
--train-path data/wiki5m_ind/train.txt.json \
--valid-path data/wiki5m_ind/valid.txt.json \
--rel-path data/wiki5m_ind/relations.json \
--num-rels 822 \
--task wiki5m_ind \
--batch-size 1024 \
--print-freq 20 \
--additive-margin 0.02 \
--use-amp \
--use-self-negative \
--finetune-t \
--epochs 1 \
--workers 4 \
--max-to-keep 5 \
--prefix-seq-len 50