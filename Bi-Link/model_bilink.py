from abc import ABC
from copy import deepcopy
import torch
import torch.nn as nn

from dataclasses import dataclass
from transformers import AutoModel, AutoConfig

latency = 0
def build_model(args) -> nn.Module:
    if args.pretrained_model == 'bert-base-uncased':
        return CustomBertModel(args)
    elif args.pretrained_model == 'gpt2':
        return CustomGPTModel(args)
    else:
        raise NotImplementedError

@dataclass
class ModelOutput:
    logits: torch.tensor
    labels: torch.tensor
    inv_t: torch.tensor
    hr_vector: torch.tensor
    tail_vector: torch.tensor


class PrefixEncoder(torch.nn.Module):
    '''
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''
    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding((config.num_rel*2+1)*config.pre_seq_len, config.hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, config.prefix_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(config.prefix_hidden_size, config.num_hidden_layers * 2 * config.hidden_size)
            )
        else:
            self.embedding = torch.nn.Embedding((config.num_rel*2+1)*config.pre_seq_len, config.num_hidden_layers * 2 * config.hidden_size)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values


class CustomBertModel(nn.Module, ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.pretrained_model)
        self.config.num_rel = args.num_rels
        self.args.pooling = "cls"
        self.log_inv_t = torch.nn.Parameter(torch.tensor(1.0 / args.t).log(), requires_grad=args.finetune_t)
        self.add_margin = args.additive_margin
        self.batch_size = args.batch_size

        self.hr_bert = AutoModel.from_pretrained(args.pretrained_model)
        self.tail_bert = deepcopy(self.hr_bert)

        self.pre_seq_len = self.args.prefix_seq_len
        print("prefix length", self.pre_seq_len)
        self.relational_tokens = nn.Parameter(torch.arange((args.num_rels*2+1) * self.pre_seq_len).long(), requires_grad=False)
        self.prefix_tokens = self.relational_tokens[-self.pre_seq_len:]

        self.config.pre_seq_len = self.pre_seq_len
        self.config.prefix_hidden_size = self.config.hidden_size
        self.config.prefix_projection = self.args.prefix_projection
        self.prefix_encoder = PrefixEncoder(self.config)
        self.dropout = torch.nn.Dropout(self.config.hidden_dropout_prob)

        print(self.prefix_encoder)

        all_param = 0
        for _, param in self.named_parameters():
            if param.requires_grad:
                all_param += param.numel()

        print("Number of training parameters: {}".format(all_param))

    def get_prompt(self, batch_size, prefix_tokens=None, relation_ids=None):
        """Get prompt tokens to meet input requirements.

        Args:
            batch_size (_type_): _description_

        Returns:
            _type_: _description_
        """
        if relation_ids is not None:
            prefix_tokens = self.relational_tokens.view(-1, self.pre_seq_len)[relation_ids, :].to(self.hr_bert.device)
        else:
            prefix_tokens = prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.hr_bert.device)
        past_key_values = self.prefix_encoder(prefix_tokens)

        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.config.num_hidden_layers * 2, 
            self.config.num_attention_heads,
            self.config.hidden_size // self.config.num_attention_heads
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2) # layers, bs, num_heads, seq_len, hid_dim
        return past_key_values, prefix_tokens
    
    def _encode(self, encoder, token_ids, mask, token_type_ids, past_key_values):
        prefix_mask = torch.ones(len(mask), self.pre_seq_len).long().to(mask.device)
        prefix_mask = torch.cat([prefix_mask, mask], dim=1)
        # attend to past key values
        outputs = encoder(input_ids=token_ids,
                          attention_mask=prefix_mask,
                          token_type_ids=token_type_ids,
                          past_key_values=past_key_values,)[0]
        cls_outputs = outputs[:, 0, :]
        return cls_outputs

    def forward(self, hr_token_ids, hr_mask, hr_token_type_ids,
                tail_token_ids, tail_mask, tail_token_type_ids,
                head_token_ids, head_mask, head_token_type_ids,
                relation_ids,
                only_ent_embedding=False, **kwargs) -> dict:
        batchsize = len(hr_token_ids)

        tail_past_key_values, prefix_tokens = self.get_prompt(batchsize, self.prefix_tokens)
        if only_ent_embedding:
            return self.predict_ent_embedding(tail_token_ids=tail_token_ids,
                                              tail_mask=tail_mask,
                                              tail_token_type_ids=tail_token_type_ids,
                                              past_key_values=tail_past_key_values
                                              )

        hr_past_key_values, prefix_tokens = self.get_prompt(batchsize, relation_ids=relation_ids)

        hr_vector = self._encode(self.hr_bert,
                                 token_ids=hr_token_ids,
                                 mask=hr_mask,
                                 token_type_ids=hr_token_type_ids,
                                 past_key_values=hr_past_key_values
                                 )

        tail_vector = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids,
                                   past_key_values=tail_past_key_values
                                   )

        head_vector = self._encode(self.tail_bert,
                                   token_ids=head_token_ids,
                                   mask=head_mask,
                                   token_type_ids=head_token_type_ids,
                                   past_key_values=tail_past_key_values
                                   )
        # DataParallel only support tensor/dict
        return {'hr_vector': hr_vector,
                'tail_vector': tail_vector,
                'head_vector': head_vector}

    def compute_logits(self, output_dict: dict, batch_dict: dict) -> dict:
        hr_vector, tail_vector = output_dict['hr_vector'], output_dict['tail_vector']
        batch_size = hr_vector.size(0)
        labels = torch.arange(batch_size).to(hr_vector.device)

        logits = hr_vector.mm(tail_vector.t())
        if self.training:
            logits -= torch.zeros(logits.size()).fill_diagonal_(self.add_margin).to(logits.device)
        logits *= self.log_inv_t.exp()

        triplet_mask = batch_dict.get('triplet_mask', None)
        if triplet_mask is not None:
            logits.masked_fill_(~triplet_mask, -1e4)

        if self.args.use_self_negative and self.training:
            head_vector = output_dict['head_vector']
            self_neg_logits = torch.sum(hr_vector * head_vector, dim=1) * self.log_inv_t.exp()
            self_negative_mask = batch_dict['self_negative_mask']
            self_neg_logits.masked_fill_(~self_negative_mask, -1e4)
            logits = torch.cat([logits, self_neg_logits.unsqueeze(1)], dim=-1)

        return {'logits': logits,
                'labels': labels,
                'inv_t': self.log_inv_t.detach().exp(),
                'hr_vector': hr_vector.detach(),
                'tail_vector': tail_vector.detach()}


    @torch.no_grad()
    def predict_ent_embedding(self, tail_token_ids, tail_mask, tail_token_type_ids, past_key_values, **kwargs) -> dict:
        ent_vectors = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids,
                                   past_key_values=past_key_values,
                                   )
        return {'ent_vectors': ent_vectors.detach()}


class CustomGPTModel(nn.Module, ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.pretrained_model)
        self.config.num_rel = args.num_rels
        self.args.pooling = "cls"
        self.log_inv_t = torch.nn.Parameter(torch.tensor(1.0 / args.t).log(), requires_grad=args.finetune_t)
        self.add_margin = args.additive_margin
        self.batch_size = args.batch_size

        self.hr_gpt = AutoModel.from_pretrained(args.pretrained_model)
        self.tail_gpt = deepcopy(self.hr_gpt)

        self.pre_seq_len = self.args.prefix_seq_len
        print("prefix length", self.pre_seq_len)
        self.relational_tokens = nn.Parameter(torch.arange((args.num_rels * 2 + 1) * self.pre_seq_len).long(), requires_grad=False)
        self.prefix_tokens = self.relational_tokens[-self.pre_seq_len:]

        self.config.pre_seq_len = self.pre_seq_len
        self.config.prefix_hidden_size = self.config.hidden_size
        self.config.prefix_projection = self.args.prefix_projection
        self.prefix_encoder = PrefixEncoder(self.config)
        self.dropout = torch.nn.Dropout(0.1)

        print(self.prefix_encoder)

        all_param = 0
        for _, param in self.named_parameters():
            if param.requires_grad:
                all_param += param.numel()

        print("Number of training parameters: {}".format(all_param))

    def get_prompt(self, batch_size, prefix_tokens=None, relation_ids=None):
        """Get prompt tokens to meet input requirements.

        Args:
            batch_size (_type_): _description_

        Returns:
            _type_: _description_
        """
        if relation_ids is not None:
            prefix_tokens = self.relational_tokens.view(-1, self.pre_seq_len)[relation_ids, :].to(self.hr_gpt.device)
        else:
            prefix_tokens = prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.hr_gpt.device)
        past_key_values = self.prefix_encoder(prefix_tokens)

        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.config.num_hidden_layers * 2,
            self.config.num_attention_heads,
            self.config.hidden_size // self.config.num_attention_heads
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)  # layers, bs, num_heads, seq_len, hid_dim
        return past_key_values, prefix_tokens

    def _encode(self, encoder, token_ids, mask, token_type_ids, past_key_values):
        prefix_mask = torch.ones(len(mask), self.pre_seq_len).long().to(mask.device)
        prefix_mask = torch.cat([prefix_mask, mask], dim=1)
        # attend to past key values
        outputs = encoder(input_ids=token_ids,
                          attention_mask=prefix_mask,
                          token_type_ids=token_type_ids,
                          past_key_values=past_key_values, )[0]
        last_outputs = outputs[:, -1, :]
        return last_outputs

    def forward(self, hr_token_ids, hr_mask, hr_token_type_ids,
                tail_token_ids, tail_mask, tail_token_type_ids,
                head_token_ids, head_mask, head_token_type_ids,
                relation_ids,
                only_ent_embedding=False, **kwargs) -> dict:
        batchsize = len(hr_token_ids)
        # global latency
        tail_past_key_values, prefix_tokens = self.get_prompt(batchsize, self.prefix_tokens)
        if only_ent_embedding:
            return self.predict_ent_embedding(tail_token_ids=tail_token_ids,
                                              tail_mask=tail_mask,
                                              tail_token_type_ids=tail_token_type_ids,
                                              past_key_values=tail_past_key_values
                                              )

        hr_past_key_values, prefix_tokens = self.get_prompt(batchsize, relation_ids=relation_ids)

        hr_vector = self._encode(self.hr_gpt,
                                 token_ids=hr_token_ids,
                                 mask=hr_mask,
                                 token_type_ids=hr_token_type_ids,
                                 past_key_values=hr_past_key_values
                                 )

        tail_vector = self._encode(self.tail_gpt,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids,
                                   past_key_values=tail_past_key_values
                                   )

        head_vector = self._encode(self.tail_gpt,
                                   token_ids=head_token_ids,
                                   mask=head_mask,
                                   token_type_ids=head_token_type_ids,
                                   past_key_values=tail_past_key_values
                                   )
        return {'hr_vector': hr_vector,
                'tail_vector': tail_vector,
                'head_vector': head_vector}

    def compute_logits(self, output_dict: dict, batch_dict: dict) -> dict:
        hr_vector, tail_vector = output_dict['hr_vector'], output_dict['tail_vector']
        batch_size = hr_vector.size(0)
        labels = torch.arange(batch_size).to(hr_vector.device)

        logits = hr_vector.mm(tail_vector.t())
        if self.training:
            logits -= torch.zeros(logits.size()).fill_diagonal_(self.add_margin).to(logits.device)
        logits *= self.log_inv_t.exp()

        triplet_mask = batch_dict.get('triplet_mask', None)
        if triplet_mask is not None:
            logits.masked_fill_(~triplet_mask, -1e4)

        if self.args.use_self_negative and self.training:
            head_vector = output_dict['head_vector']
            self_neg_logits = torch.sum(hr_vector * head_vector, dim=1) * self.log_inv_t.exp()
            self_negative_mask = batch_dict['self_negative_mask']
            self_neg_logits.masked_fill_(~self_negative_mask, -1e4)
            logits = torch.cat([logits, self_neg_logits.unsqueeze(1)], dim=-1)

        return {'logits': logits,
                'labels': labels,
                'inv_t': self.log_inv_t.detach().exp(),
                'hr_vector': hr_vector.detach(),
                'tail_vector': tail_vector.detach()}

    @torch.no_grad()
    def predict_ent_embedding(self, tail_token_ids, tail_mask, tail_token_type_ids, past_key_values, **kwargs) -> dict:
        ent_vectors = self._encode(self.tail_gpt,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids,
                                   past_key_values=past_key_values,
                                   )
        return {'ent_vectors': ent_vectors.detach()}



