from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import tensor
import collections
import json
import math
import random
from transformers import AutoConfig, AutoTokenizer
import json
from dataclasses import dataclass, field
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
from utils import logger
from mention_mask import get_mention_mask, get_label_mask

@dataclass
class CLInstance(object):
    """A single set of features for EL as a contrastive learning task"""
    mention_dict: Dict
    entity_dict: Dict
    candidate_dicts: List[Dict]
    mention_id: str
    label_id: str
    label: tensor
    corpus: str

def load_candidates(input_dir):
    documents = {}
    with open(input_dir, "r", encoding="utf-8") as reader:
        doc_dicts = json.load(reader)

    for doc in doc_dicts:
        documents[doc['mention_id']] = doc["tfidf_candidates"]  # mention_id to candidates

    return documents

def load_documents(input_dir):
    documents = {}
    with open(input_dir, "r", encoding="utf-8") as reader:
        while True:
            line = reader.readline()
            line = line.strip()
            if not line:
                break
            line = json.loads(line)
            documents[line['document_id']] = line
    logger.info("Loading {} documents from {}".format(len(documents), input_dir))
    return documents

def get_context_tokens(tokenizer, context_tokens, start_index, end_index, max_tokens):
    " extract mention context with an evenly distributed sliding window"
    start_pos = start_index - max_tokens
    if start_pos < 0:
        start_pos = 0
    prefix = ' '.join(context_tokens[start_pos: start_index])
    suffix = ' '.join(context_tokens[end_index + 1: end_index + max_tokens + 1])
    prefix = tokenizer.tokenize(prefix)
    suffix = tokenizer.tokenize(suffix)
    mention = tokenizer.tokenize(' '.join(context_tokens[start_index: end_index + 1]))

    assert len(mention) < max_tokens

    remaining_tokens = max_tokens - len(mention)
    half_remaining_tokens = int(math.ceil(1.0 * remaining_tokens / 2))

    # protect the shorter side
    if len(prefix) >= half_remaining_tokens and len(suffix) >= half_remaining_tokens:
        prefix_len = half_remaining_tokens
    elif len(prefix) >= half_remaining_tokens and len(suffix) < half_remaining_tokens:
        prefix_len = remaining_tokens - len(suffix)
        if prefix_len > len(prefix):
            prefix_len = len(prefix)
    elif len(prefix) < half_remaining_tokens:
        prefix_len = len(prefix)
    else:
        raise ValueError

    prefix = prefix[-prefix_len:]

    mention_context = prefix + mention + suffix
    mention_start = len(prefix)
    mention_end = mention_start + len(mention)
    mention_context = mention_context[:max_tokens]

    assert mention_start <= max_tokens
    assert mention_end <= max_tokens

    return mention_context, mention_start, mention_end

def pad_sequence(tokens, max_len):
    assert len(tokens) <= max_len
    return tokens + [0]*(max_len - len(tokens))

def customized_tokenize(tokenizer, tokens, max_seq_length, mention_start=None,
                        mention_end=None, return_tensor="pt"):
    if type(tokens) ==str:
        tokens = tokenizer.tokenize(tokens, add_special_tokens=True, max_length=max_seq_length, truncation=True)
    else:
        tokens = ["[CLS]"] + tokens + ["[SEP]"]

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # token type ids
    token_type_ids = [0]*len(tokens)
    if mention_start and mention_end:
        for idx in range(mention_start+1, mention_end+1):
            token_type_ids[idx] = 2
    #  set mention span as 2 ["CLS"]
    attention_mask = [1]*len(input_ids)

    input_ids = pad_sequence(input_ids, max_seq_length)
    token_type_ids = pad_sequence(token_type_ids, max_seq_length)
    attention_mask = pad_sequence(attention_mask, max_seq_length)

    if return_tensor == "pt":
        input_ids = torch.LongTensor([input_ids])
        token_type_ids = torch.LongTensor([token_type_ids])
        attention_mask = torch.LongTensor([attention_mask])
    return {"input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask
            }

class EntityLinkingSet(Dataset):
    """Create `TrainingInstance`s from raw text."""
    def __init__(self, pretrained_model_path, document_files, mentions_path, tfidf_candidates_file, max_seq_length,
                 num_candidates, is_training=True,):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
        self.num_candidates = num_candidates
        self.rng = random.Random(12345)
        self.max_seq_length = max_seq_length
        self.is_training = is_training
        # self.document_path = document_files[0].split(",")
        # self.all_documents = {}      # doc_id/ entity_id to entity
        #
        # for input_file_path in self.document_path:
        #     self.all_documents.update(load_documents(input_file_path))
        self.all_documents = document_files
        self.candidates = load_candidates(tfidf_candidates_file)   # mention_id to candidates

        self.mentions = self.load_mentions(mentions_path)       # mention_id, context_id, label_id, start_idx, end_idx


    def filter_mention(self, mention):
        mention_id = mention["mention_id"]
        label_document_id = mention["label_document_id"]
        assert mention_id in self.candidates
        cand_document_ids = self.candidates[mention_id]
        # skip this mention if there is no tf-idf candidate
        if not cand_document_ids:
            return None
        # if manually labelled description doc of the mention is not in the noisy tf-idf set, skip this mention
        elif not self.is_training and label_document_id not in cand_document_ids:
            return None
        else:
            return mention

    def load_mentions(self, mention_dir):
        with open(mention_dir, "r", encoding="utf-8") as m:
            mentions = json.load(m)
        logger.info("Loading {} mentions from {}".format(len(mentions), mention_dir))
        return mentions

    def reserve_topk_tf_idf_candidates(self):
        if not self.is_training:
            topk_candidates_dict = {}
            for cand_key, cand_document_ids in self.candidates.items():
                topk_candidates_dict[cand_key] = cand_document_ids[:self.num_candidates]

            self.candidates = topk_candidates_dict


    def __len__(self):
        return len(self.mentions)

    def __getitem__(self, item):
        return self.create_cl_instances(self.mentions[item])

    def create_cl_instances(self, mention):
        """Creates Next Sentence Prediction Instance for a single document."""

        # Account for [CLS], [SEP]
        max_num_tokens = self.max_seq_length - 2

        # mention and context
        mention_id = mention['mention_id']
        context_document_id = mention['context_document_id']
        label_document_id = mention['label_document_id']
        start_index = mention['start_index']  # start idx in the context doc
        end_index = mention['end_index']  # end idx in the context doc

        context = self.all_documents[context_document_id]['text']

        context_tokens = context.split()
        extracted_mention = context_tokens[start_index: end_index + 1]
        extracted_mention = ' '.join(extracted_mention)
        mention_text = mention['text']
        assert extracted_mention == mention_text

        mention_context, mention_start, mention_end = get_context_tokens(self.tokenizer,
            context_tokens, start_index, end_index, max_num_tokens)

        label_idx = mention['label']        # indices of gts in candidate sets
        input_dicts = customized_tokenize(self.tokenizer, mention_context, self.max_seq_length, mention_start, mention_end)

        label_document = self.all_documents[label_document_id]['text']
        label_dicts = customized_tokenize(self.tokenizer, label_document, self.max_seq_length)

        # adding tf-idf candidates (including gt by default)
        cand_document_ids = self.candidates[mention_id]
        if self.is_training:
            # del gt from negative samples
            # cand_document_ids = [cand for cand in cand_document_ids if cand != label_document_id]
            del cand_document_ids[label_idx]

            cand_document_ids = cand_document_ids[:self.num_candidates]     # truncate to num_candidates for cl

        candidates_input_dicts = []
        for cand_document_id in cand_document_ids:
            cand_document = self.all_documents[cand_document_id]['text']
            cand_dict = customized_tokenize(self.tokenizer, cand_document, self.max_seq_length)

            candidates_input_dicts.append(cand_dict)

        instance = CLInstance(
            mention_dict=input_dicts,
            entity_dict=label_dicts,
            candidate_dicts=candidates_input_dicts,
            mention_id=mention_id,
            label_id=label_document_id,
            label=torch.LongTensor([label_idx]),
            corpus=mention['corpus']
        )
        return instance

def collate(batch_data):
    input_ids = []
    attention_mask = []
    token_type_ids = []

    for sep_dict in batch_data:
        input_ids.append(sep_dict['input_ids'])
        attention_mask.append(sep_dict['attention_mask'])
        token_type_ids.append(sep_dict['token_type_ids'])

    return {"input_ids": torch.cat(input_ids, 0),
            "attention_mask": torch.cat(attention_mask, 0),
            "token_type_ids": torch.cat(token_type_ids, 0),
            }


def compose_collate(batch_cl_data: List[CLInstance]):
    mention_dicts = [cl_data.mention_dict for cl_data in batch_cl_data]
    mention_dicts = collate(mention_dicts)

    label_dicts = [cl_data.entity_dict for cl_data in batch_cl_data]
    label_dicts = collate(label_dicts)

    labels = [cl_data.label for cl_data in batch_cl_data]
    labels = torch.cat(labels, 0)

    mention_ids = [cl_data.mention_id for cl_data in batch_cl_data]
    label_ids = [cl_data.label_id for cl_data in batch_cl_data]

    input_ids = []
    attention_mask = []
    token_type_ids = []

    for cl_data in batch_cl_data:
        cand_dict_list = cl_data.candidate_dicts
        _cand_dict = collate(cand_dict_list)
        input_ids.append(_cand_dict['input_ids'])
        attention_mask.append(_cand_dict['attention_mask'])
        token_type_ids.append(_cand_dict['token_type_ids'])

    return {
        "mention_dicts": mention_dicts,
        "entity_dicts": label_dicts,
        "labels": labels,
        "me_mask": get_label_mask(mention_ids, label_ids),
        "mm_mask": get_mention_mask(mention_ids),
        "candidate_dicts": {"input_ids": torch.stack(input_ids, 0),
                            "attention_mask": torch.stack(attention_mask, 0),
                            "token_type_ids": torch.stack(token_type_ids, 0)
                            }
    }
