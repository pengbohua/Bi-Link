from lib2to3.pgen2 import token
import os
import json
import torch
import torch.utils.data.dataset

from typing import Optional, List, Tuple

from config import args
from triplet_mask import construct_mask, construct_self_negative_mask
from dict_hub import get_entity_dict, get_link_graph, get_tokenizer
from logger_config import logger
import logging


entity_dict = get_entity_dict()
if args.use_link_graph:
    # make the lazy data loading happen
    get_link_graph()

tokenizer = get_tokenizer()
SEP_TOKEN = tokenizer.sep_token
SEP = tokenizer.convert_tokens_to_ids(SEP_TOKEN)
mask_token = tokenizer.mask_token

logger = logging.getLogger()


def _parse_entity_name(entity: str) -> str:
    if args.task.lower() == 'wn18rr':
        # family_alcidae_NN_1
        entity = ' '.join(entity.split('_')[:-2])
        return entity
    # a very small fraction of entities in wiki5m do not have name
    return entity or ''


def _concat_name_desc(entity: str, entity_desc: str) -> str:
    if entity_desc.startswith(entity):
        entity_desc = entity_desc[len(entity):].strip()
    if entity_desc:
        return '{} {}'.format(entity, entity_desc)
    return entity

def _get_desc_text(entity: str, entity_desc: str) -> str:
    if entity_desc.startswith(entity):
        entity_desc = entity_desc[len(entity):]
    # filtering
    # entity_desc = entity_desc.replace('"', '')
    return entity_desc

def get_neighbor_desc(head_id: str, tail_id: str = None) -> str:
    neighbor_ids = get_link_graph().get_neighbor_ids(head_id)
    # avoid label leakage during training
    if not args.is_test:
        neighbor_ids = [n_id for n_id in neighbor_ids if n_id != tail_id]
    entities = [entity_dict.get_entity_by_id(n_id).entity for n_id in neighbor_ids]
    entities = [_parse_entity_name(entity) for entity in entities]
    return ' '.join(entities)


def pad_sequence(tokens, max_len):
    assert len(tokens) <= max_len, "token length: {}".format(len(tokens))
    return tokens + [0] * (max_len - len(tokens))

def get_template_token_ids(tokenizer, template):
    template_list = template.split()
    template_list.remove("[X]")
    template_list.remove("[REL]")
    _template = " ".join(template_list)
    _template = tokenizer.tokenize(text=_template)+[mask_token]
    template_tokens = tokenizer.convert_tokens_to_ids(_template)
    return torch.LongTensor([template_tokens])

def customized_tokenize(tokenizer, 
                        text_entity, 
                        text_entity_desc, 
                        text_rel=None, text_rel_max_len=3, 
                        max_seq_length=50,
                        forward=True,
                        template = None,
                        template_position_ids = None,
                        return_tensor="pt"):

    if template is not None:    
        template_tokens = get_template_token_ids(tokenizer, template)
        head_idx = template.split().index("[X]")
        rel_idx = template.split().index("[REL]")
        template_mid = " ".join(template.split()[head_idx + 1: rel_idx])
        template_after_rel = " ".join(template.split()[rel_idx + 1:])
        token_entity = tokenizer.tokenize(text=text_entity)
        token_entity_desc = tokenizer.tokenize(text=text_entity_desc)
        token_pair_rel = tokenizer.tokenize(text=text_rel)[:text_rel_max_len]
        template_mid = tokenizer.tokenize(text=template_mid)
        template_after_rel = tokenizer.tokenize(text=template_after_rel)

        template_len = len(template_mid) + len(template_after_rel)  # HEAD, Prompt, REL, Prompt + [MASK]

        max_seq_a = max_seq_length - template_len - len(token_entity) - len(token_pair_rel) - 1
        token_entity_desc = token_entity_desc[:max_seq_a]

        if forward:
            tokens = token_entity_desc + token_entity + template_mid + token_pair_rel + template_after_rel + [mask_token]    # DESC + HEAD, Prompt, REL, Prompt + [MASK] 
            token_type_ids = [0] * (len(token_entity_desc) + len(token_entity) + len(template_mid)) + [1] * len(token_pair_rel) + [0] * (len(template_after_rel) + 1)
            idx_rel_end = len(token_entity + template_mid + token_pair_rel)
            idx_mask = len(token_entity + template_mid + token_pair_rel + template)
            pos_range = torch.arange(len(tokens))       # template denoising
            template_position_ids = torch.cat([pos_range[len(token_entity):len(token_entity + template_mid)], pos_range[idx_rel_end:idx_mask]]).long()

        else:
            tokens = [mask_token] + template_mid + token_pair_rel + template_after_rel + token_entity + token_entity_desc   # [MASK], Prompt, REL, Prompt + TAIL + DESC
            token_type_ids = [0] * (1 + len(template_mid)) + [1] * len(token_pair_rel) + [0] * (len(template_after_rel) + len(token_entity) + len(token_entity_desc))            
            idx_rel_end = 1 + len(template_mid + token_pair_rel)
            idx_desc = 1 + len(template_mid + token_pair_rel + template_after_rel)
            pos_range = torch.arange(len(tokens))       # template denoising
            template_position_ids = torch.cat([pos_range[1:1+len(template_mid)], pos_range[idx_rel_end:idx_desc]]).long()
        
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)


    elif text_entity is not None and text_rel is not None:
        token_entity = tokenizer.tokenize(text=text_entity)     
        token_entity_desc = tokenizer.tokenize(text_entity_desc)  
        token_pair_rel = tokenizer.tokenize(text=text_rel)[:text_rel_max_len]
        max_seq_a = max_seq_length - len(token_entity) - len(token_pair_rel) - 1

        if forward:
            tokens = token_entity + token_entity_desc[:max_seq_a] + token_pair_rel + [mask_token]     # DESC + HEAD, REL, [MASK]
            token_type_ids = [0]*(len(token_entity) + max_seq_a) + [1]*len(token_pair_rel) + [0]
        else:
            tokens = [mask_token] + token_pair_rel + token_entity + token_entity_desc[:max_seq_a]  # [MASK], REL, HEAD + DESC
            token_type_ids = [0] + [1]*len(token_pair_rel) + [0]*(max_seq_a+len(token_entity))      

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
    elif text_entity is not None and text_rel is None:   # [MASK], ENT, DESC
        token_entity = tokenizer.tokenize(text=text_entity)     
        token_entity_desc = tokenizer.tokenize(text_entity_desc) 
        max_seq_a = max_seq_length - len(token_entity) - 1

        tokens = [mask_token] + token_entity + token_entity_desc[:max_seq_a]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        token_type_ids = [0]*len(input_ids)
        attention_mask = [1] * len(input_ids)
    else:
        raise ValueError("inappropriate relational inputs")

    input_ids = pad_sequence(input_ids, max_seq_length)
    token_type_ids = pad_sequence(token_type_ids, max_seq_length)
    attention_mask = pad_sequence(attention_mask, max_seq_length)

    if return_tensor == "pt":
        input_ids = torch.LongTensor([input_ids])
        token_type_ids = torch.LongTensor([token_type_ids])
        attention_mask = torch.LongTensor([attention_mask])
    return {"input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "template": template_tokens if template else None,
            "template_position_ids": template_position_ids if template else None
            }

class Example:
    "A basic class for a contextualized triple"

    def __init__(self, head_id, relation, tail_id, template=None, **kwargs):
        self.template = template
        self.head_id = head_id
        self.tail_id = tail_id
        self.relation = relation

    @property
    def head_desc(self):
        if not self.head_id:
            return ''
        return entity_dict.get_entity_by_id(self.head_id).entity_desc

    @property
    def tail_desc(self):
        if not self.tail_id:
            return ''
        return entity_dict.get_entity_by_id(self.tail_id).entity_desc

    @property
    def head(self):
        if not self.head_id:
            return ''
        return entity_dict.get_entity_by_id(self.head_id).entity

    @property
    def tail(self):
        if not self.tail_id:
            return ''
        return entity_dict.get_entity_by_id(self.tail_id).entity

    def vectorize(self) -> dict:
        head_desc, tail_desc = self.head_desc, self.tail_desc
        if args.use_link_graph:
            # 如果文本长度不够从邻居补全到20
            if len(head_desc.split()) < 20:
                head_desc += ' ' + get_neighbor_desc(head_id=self.head_id, tail_id=self.tail_id)
            if len(tail_desc.split()) < 20:
                tail_desc += ' ' + get_neighbor_desc(head_id=self.tail_id, tail_id=self.head_id)

        # a small fraction of entities wiki5m do not have names
        head_word = _parse_entity_name(self.head)
        head_desc = _get_desc_text(head_word, head_desc)

        hr_encoded_inputs = customized_tokenize(tokenizer,
                                                text_entity=head_word,
                                                text_entity_desc=head_desc,
                                                text_rel=self.relation,
                                                text_rel_max_len=3,
                                                max_seq_length=50,
                                                template=self.template
                                                )

        head_encoded_inputs = customized_tokenize(
                                                tokenizer,
                                                text_entity=head_word,
                                                text_entity_desc=head_desc,
                                                max_seq_length=50,
                                               )

        tail_word = _parse_entity_name(self.tail)
        tail_desc = _get_desc_text(tail_word, tail_desc)

        tr_encoded_inputs = customized_tokenize(tokenizer,
                                                text_entity=tail_word,
                                                text_entity_desc=tail_desc,
                                                text_rel=self.relation,
                                                text_rel_max_len=3,
                                                forward=False,
                                                template=self.template
                                               )

        tail_encoded_inputs = customized_tokenize(
                                               tokenizer,
                                               text_entity=tail_word,
                                               text_entity_desc=tail_desc,
                                               max_seq_length=50,
                                               )
                                               
        return {'hr_token_ids': hr_encoded_inputs['input_ids'],
                'hr_token_mask': hr_encoded_inputs['attention_mask'],
                'hr_token_type_ids': hr_encoded_inputs['token_type_ids'],

                'head_token_ids': head_encoded_inputs['input_ids'],
                'head_token_mask': head_encoded_inputs['attention_mask'],
                'head_token_type_ids': head_encoded_inputs['token_type_ids'],

                'tr_token_ids': tr_encoded_inputs['input_ids'],
                'tr_token_mask': tr_encoded_inputs['attention_mask'],
                'tr_token_type_ids': tr_encoded_inputs['token_type_ids'],

                'tail_token_ids': tail_encoded_inputs['input_ids'],
                'tail_token_mask': tail_encoded_inputs['attention_mask'],                
                'tail_token_type_ids': tail_encoded_inputs['token_type_ids'],

                'template': hr_encoded_inputs['template'] if 'template' in hr_encoded_inputs.keys() else None,
                'left_template_position_ids': hr_encoded_inputs['template_position_ids'] if 'template_position_ids' in hr_encoded_inputs.keys() else None,
                'right_template_position_ids': tr_encoded_inputs['template_position_ids'] if 'template_position_ids' in tr_encoded_inputs.keys() else None,
                'obj': self}


class Dataset(torch.utils.data.dataset.Dataset):

    def __init__(self, path, task, examples=None):
        self.path_list = path.split(',')
        self.task = task
        assert all(os.path.exists(path) for path in self.path_list) or examples
        if examples:
            self.examples = examples
        else:
            self.examples = []
            for path in self.path_list:
                if not self.examples:
                    # load data with template
                    self.examples = load_data(path)
                else:
                    self.examples.extend(load_data(path))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index].vectorize()

def load_vis_data(path: str,
              add_forward_triplet: bool = True,
              add_backward_triplet: bool = True) -> List[Example]:
    assert path.endswith('.json'), 'Unsupported format: {}'.format(path)
    assert add_forward_triplet or add_backward_triplet
    logger.info('In test mode: {}'.format(args.is_test))

    data = json.load(open(path, 'r', encoding='utf-8'))
    logger.info('Load {} examples from {}'.format(len(data), path))

    cnt = len(data)
    examples = []
    for i in range(cnt):
        obj = data[i]
        if add_forward_triplet:
            examples.append(Example(**obj))
        if add_backward_triplet:
            examples.append(Example(**reverse_triplet(obj)))
        data[i] = None

    return examples

class VisRelExpDataset(Dataset):
    # Dataset for relational expression visualization
    def __init__(self, path, task, examples=None):
        super(VisRelExpDataset, self).__init__(path=path, task=task, examples=examples)
        self.path_list = path.split(',')
        self.task = task
        assert all(os.path.exists(path) for path in self.path_list) or examples
        if examples:
            self.examples = examples
        else:
            self.examples = []
            for path in self.path_list:
                if not self.examples:
                    # load data without template and "inverse"
                    self.examples = load_vis_data(path)
                else:
                    self.examples.extend(load_vis_data(path))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index].vectorize_hr()

def load_data(path: str,
              add_forward_triplet: bool = True,
              add_backward_triplet: bool = False) \
        -> List[Example]:
    assert path.endswith('.json'), 'Unsupported format: {}'.format(path)
    assert add_forward_triplet or add_backward_triplet
    logger.info('In test mode: {}'.format(args.is_test))

    # template
    template_hr = None
    template_hr_inv = None
    # template_hr = '[X] has [REL] as'
    # template_hr_inv = '[X] is [REL] of'

    # [{tiple_1}, {triple_2}, {triple_n}]
    data = json.load(open(path, 'r', encoding='utf-8'))
    logger.info('Load {} examples from {}'.format(len(data), path))

    cnt = len(data)
    examples = []
    for i in range(cnt):
        obj = data[i]
        if add_forward_triplet:
            examples.append(Example(template=template_hr, **obj))
        if add_backward_triplet:
            # 数据增强 加上反向关系，inverse published by
            examples.append(Example(template=template_hr_inv, **reverse_triplet(obj)))
        data[i] = None

    return examples


def collate(batch_data: List[dict]) -> dict:
    hr_token_ids = []
    hr_mask = []
    hr_token_type_ids = []

    head_token_ids = []
    head_entity_mask = []
    head_token_type_ids = []

    tr_token_ids = []
    tr_mask = []
    tr_token_type_ids = []

    tail_token_ids = []
    tail_entity_mask = []
    tail_token_type_ids = []

    batch_exs = []
    templates = []
    left_template_position_ids = []
    right_template_position_ids = []

    for data_dict in batch_data:
        hr_token_ids.append(data_dict['hr_token_ids'])  # batch_size, 50
        hr_mask.append(data_dict['hr_token_mask'])
        hr_token_type_ids.append(data_dict['hr_token_type_ids'])

        head_token_ids.append(data_dict['head_token_ids'])  
        head_entity_mask.append(data_dict['head_token_mask'])
        head_token_type_ids.append(data_dict['head_token_type_ids'])

        tr_token_ids.append(data_dict['tr_token_ids'])
        tr_mask.append(data_dict['tr_token_mask'])
        tr_token_type_ids.append(data_dict['tr_token_type_ids'])

        tail_token_ids.append(data_dict['tail_token_ids'])
        tail_entity_mask.append(data_dict['tail_token_mask'])
        tail_token_type_ids.append(data_dict['tail_token_type_ids'])

        templates.append(data_dict['template'])
        left_template_position_ids.append(data_dict['left_template_position_ids'])
        right_template_position_ids.append(data_dict['right_template_position_ids'])
        batch_exs.append(data_dict['obj'])

    batch_dict = {
        'hr_batch_dict': {
        'input_ids': torch.cat(hr_token_ids, 0),
        'attention_mask': torch.cat(hr_mask, 0),
        'token_type_ids': torch.cat(hr_token_type_ids, 0),
        },

        'head_batch_dict': {
        'input_ids': torch.cat(head_token_ids, 0),
        'attention_mask': torch.cat(head_entity_mask, 0),
        'token_type_ids': torch.cat(head_token_type_ids, 0),
        },
        'tr_batch_dict': {
        'input_ids': torch.cat(tr_token_ids, 0),
        'attention_mask': torch.cat(tr_mask, 0),
        'token_type_ids': torch.cat(tr_token_type_ids, 0),
        },
        'tail_batch_dict': {
        'input_ids': torch.cat(tail_token_ids, 0),
        'attention_mask': torch.cat(tail_entity_mask, 0),
        'token_type_ids': torch.cat(tail_token_type_ids, 0),
        },
        'template': torch.cat(templates, 0) if data_dict['template'] is not None else None,
        'left_template_position_ids': torch.stack(left_template_position_ids, 0) if data_dict['template'] is not None else None,
        'right_template_position_ids': torch.stack(right_template_position_ids, 0) if data_dict['template'] is not None else None,

        'batch_data': batch_exs,
        'triplet_mask': construct_mask(row_exs=batch_exs) if not args.is_test else None,
        # h r other availabe tails in this batch
        'self_negative_mask': construct_self_negative_mask(batch_exs) if not args.is_test else None,
        # head relation  head
    }

    return batch_dict

def vis_collate(batch_data: List[dict]) -> dict:
    '''collate func for pretraining with a new pretraining task split point prediction'''
    input_ids = []
    attention_mask = []
    token_type_ids = []
    for data_dict in batch_data:
        input_ids.append(data_dict['input_ids'])  # batch_size, 50
        attention_mask.append(data_dict['attention_mask'])
        token_type_ids.append(data_dict['token_type_ids'])

    return {"input_ids": torch.cat(input_ids, 0),
            "attention_mask": torch.cat(attention_mask, 0),
            "token_type_ids": torch.cat(token_type_ids, 0)
            }

def eval_collate(batch_data: List[dict]) -> dict:
    hr_token_ids = []
    hr_mask = []
    hr_token_type_ids = []

    head_token_ids = []
    head_entity_mask = []
    head_token_type_ids = []

    tr_token_ids = []
    tr_mask = []
    tr_token_type_ids = []

    tail_token_ids = []
    tail_entity_mask = []
    tail_token_type_ids = []

    batch_exs = []
    templates = []
    left_template_position_ids = []
    right_template_position_ids = []

    for data_dict in batch_data:
        hr_token_ids.append(data_dict['hr_token_ids'])  # batch_size, 50
        hr_mask.append(data_dict['hr_token_mask'])
        hr_token_type_ids.append(data_dict['hr_token_type_ids'])

        head_token_ids.append(data_dict['head_token_ids'])  
        head_entity_mask.append(data_dict['head_token_mask'])
        head_token_type_ids.append(data_dict['head_token_type_ids'])

        tr_token_ids.append(data_dict['tr_token_ids'])
        tr_mask.append(data_dict['tr_token_mask'])
        tr_token_type_ids.append(data_dict['tr_token_type_ids'])

        tail_token_ids.append(data_dict['tail_token_ids'])
        tail_entity_mask.append(data_dict['tail_token_mask'])
        tail_token_type_ids.append(data_dict['tail_token_type_ids'])

        templates.append(data_dict['template'])
        left_template_position_ids.append(data_dict['left_template_position_ids'])
        right_template_position_ids.append(data_dict['right_template_position_ids'])
        batch_exs.append(data_dict['obj'])

    batch_dict = {
        'hr_batch_dict': {
        'input_ids': torch.cat(hr_token_ids, 0),
        'attention_mask': torch.cat(hr_mask, 0),
        'token_type_ids': torch.cat(hr_token_type_ids, 0),
        },

        'head_batch_dict': {
        'input_ids': torch.cat(head_token_ids, 0),
        'attention_mask': torch.cat(head_entity_mask, 0),
        'token_type_ids': torch.cat(head_token_type_ids, 0),
        },
        'tr_batch_dict': {
        'input_ids': torch.cat(tr_token_ids, 0),
        'attention_mask': torch.cat(tr_mask, 0),
        'token_type_ids': torch.cat(tr_token_type_ids, 0),
        },
        'tail_batch_dict': {
        'input_ids': torch.cat(tail_token_ids, 0),
        'attention_mask': torch.cat(tail_entity_mask, 0),
        'token_type_ids': torch.cat(tail_token_type_ids, 0),
        },
        'template': torch.cat(templates, 0) if data_dict['template'] is not None else None,
        'left_template_position_ids': torch.stack(left_template_position_ids, 0) if data_dict['template'] is not None else None,
        'right_template_position_ids': torch.stack(right_template_position_ids, 0) if data_dict['template'] is not None else None,

        'batch_data': batch_exs,
        'triplet_mask': None,  # h r other availabe tails in this batch
        'self_negative_mask': None,  # h r h
    }

    return batch_dict


def ent_collate(batch_data: List[dict]) -> dict:
    head_token_ids = []
    head_token_type_ids = []
    head_entity_mask = []

    tail_token_ids = []
    tail_token_type_ids = []
    tail_entity_mask = []

    for data_dict in batch_data:
        head_token_ids.append(data_dict['head_token_ids'])  # batch_size, 50
        head_token_type_ids.append(data_dict['head_token_type_ids'])
        head_entity_mask.append(data_dict['head_token_mask'])

        tail_token_ids.append(data_dict['tail_token_ids'])
        tail_token_type_ids.append(data_dict['tail_token_type_ids'])
        tail_entity_mask.append(data_dict['tail_token_mask'])

    batch_dict = {
        'head_batch_dict':{
        'input_ids': torch.cat(head_token_ids, 0),
        'token_type_ids': torch.cat(head_token_type_ids, 0),
        'attention_mask': torch.cat(head_entity_mask, 0),
        },
        'tail_batch_dict':{
        'input_ids': torch.cat(tail_token_ids, 0),
        'token_type_ids': torch.cat(tail_token_type_ids, 0),
        'attention_mask': torch.cat(tail_entity_mask, 0),
        }
    }
    return batch_dict

def reverse_triplet(obj):
    return {
        'head_id': obj['tail_id'],
        'head': obj['tail'],
        'relation': obj['relation'],
        'tail_id': obj['head_id'],
        'tail': obj['head']
    }
