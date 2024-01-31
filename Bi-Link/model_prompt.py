from abc import ABC
from copy import deepcopy
import torch
import torch.nn as nn
from dataclasses import dataclass
from transformers import AutoModel, AutoConfig
from dict_hub import get_tokenizer
import numpy as np


class SoftPrompt(nn.Module):
    def __init__(self,
                 wte: nn.Embedding,
                 num_rels: int = 237,
                 prompt_length: int = 40,
                 mode: str = 'fixed'):
        """appends learned embedding to

        Args:
            wte (nn.Embedding): original transformer word embedding
            prompt_length (int, optional): number of tokens for task. Defaults to 10.
            mode (str, optional): initialization select from {'random': randomly initialized,
                                   'fixed': "madeupword0000": 50261,
                                   'sampled': sampled from vocab embs}
        """
        super(SoftPrompt, self).__init__()
        self.wte = wte
        self.prompt_length = prompt_length
        self.mode = mode
        soft_prompts = self.initialize_embedding(prompt_length*(num_rels+1))
        self.prompts = nn.Parameter(soft_prompts.view((num_rels+1), prompt_length, -1))
        rel_ids = list(range(num_rels+1))
        self.tokenizer = get_tokenizer()
        self.tok_id2rel_id = {}
        for r_id in rel_ids:
            t_id = self.tokenizer.convert_tokens_to_ids([str(r_id)])
            self.tok_id2rel_id[t_id[0]] = r_id

    def initialize_embedding(self, total_prompt_len):
        """initializes learned embedding

        Args:
            same as __init__

        Returns:
            torch.float: initialized using original schemes
        """
        _vocab_size, _hidden_dim = self.wte.weight.shape
        if self.mode == 'fixed':
            return self.wte.weight[50261].expand(total_prompt_len, _hidden_dim).clone().detach()
        elif self.mode == 'random':
            return torch.nn.Embedding(total_prompt_len, _hidden_dim).weight
        elif self.mode == 'sequential':
            return self.wte.weight[:total_prompt_len].clone().detach()
        elif self.mode == 'sampled':
            _idx = np.random.choice(list(range(_vocab_size)), total_prompt_len)
            return self.wte.weight[_idx, :].clone().detach()
        else:
            raise NotImplementedError

    def forward(self, tokens):
        """run forward pass

        Args:
            tokens (torch.long): input tokens before encoding

        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """
        rel_tokens = tokens[:, self.prompt_length+1]                                    # take the column 1 to get rel tokens, CLS + rel_id + entity
        rel_tokens = [self.tok_id2rel_id[int(t_id.cpu())] for t_id in rel_tokens]
        soft_prompts = self.prompts[rel_tokens]
        assert len(tokens) == len(soft_prompts), "Make sure the tokenizer is loaded correctly."
        input_embedding = self.wte(tokens[:, self.prompt_length:])
        return torch.cat([soft_prompts, input_embedding], 1)


def build_model(args) -> nn.Module:
    return CustomBertModel(args)


@dataclass
class ModelOutput:
    logits: torch.tensor
    labels: torch.tensor
    inv_t: torch.tensor
    hr_vector: torch.tensor
    tail_vector: torch.tensor


class CustomBertModel(nn.Module, ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.pretrained_model)
        self.config.num_rel = args.num_rels
        self.args.pooling = "cls"
        self.log_inv_t = torch.tensor(1.0 / 20.0).log()
        self.add_margin = args.additive_margin
        self.batch_size = args.batch_size

        self.hr_bert = AutoModel.from_pretrained(args.pretrained_model)
        self.tail_bert = deepcopy(self.hr_bert)

        self.pre_seq_len = self.args.prefix_seq_len
        print("prompt length", self.pre_seq_len)
        self.config.pre_seq_len = self.pre_seq_len
        self.soft_prompts = SoftPrompt(self.tail_bert.embeddings.word_embeddings, args.num_rels, self.pre_seq_len, mode='sequential')

        for hr_param, tail_param in zip(self.hr_bert.parameters(), self.tail_bert.parameters()):
            hr_param.requires_grad = False
            tail_param.requires_grad = False

        self.hr_bert.set_input_embeddings(self.soft_prompts)
        self.tail_bert.set_input_embeddings(self.soft_prompts)

        all_param = 0
        for _, param in self.named_parameters():
            if param.requires_grad:
                all_param += param.numel()

        print("Number of training parameters: {}M".format(all_param / 1000000))

    def _encode(self, encoder, token_ids, mask, token_type_ids):
        # apply prompt tuning
        batch_size = len(mask)
        token_ids = torch.cat([torch.full((batch_size, self.pre_seq_len), 50256, device=mask.device), token_ids], dim=1)
        mask = torch.cat([torch.full((batch_size, self.pre_seq_len), 1, device=mask.device), mask], dim=1)
        token_type_ids = torch.cat([torch.full((batch_size, self.pre_seq_len), 1, device=mask.device), token_type_ids], dim=1)
        outputs = encoder(input_ids=token_ids,
                          attention_mask=mask,
                          token_type_ids=token_type_ids,
                          return_dict=True)

        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        return cls_output


    def forward(self, hr_token_ids, hr_mask, hr_token_type_ids,
                tail_token_ids, tail_mask, tail_token_type_ids,
                head_token_ids, head_mask, head_token_type_ids,
                relation_ids,
                only_ent_embedding=False, **kwargs) -> dict:
        global latency

        if only_ent_embedding:
            return self.predict_ent_embedding(tail_token_ids=tail_token_ids,
                                              tail_mask=tail_mask,
                                              tail_token_type_ids=tail_token_type_ids,
                                              )


        hr_vector = self._encode(self.hr_bert,
                                 token_ids=hr_token_ids,
                                 mask=hr_mask,
                                 token_type_ids=hr_token_type_ids,
                                 )

        tail_vector = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids,
                                   )

        head_vector = self._encode(self.tail_bert,
                                   token_ids=head_token_ids,
                                   mask=head_mask,
                                   token_type_ids=head_token_type_ids,
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
    def predict_ent_embedding(self, tail_token_ids, tail_mask, tail_token_type_ids, **kwargs) -> dict:
        ent_vectors = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids,
                                   )
        return {'ent_vectors': ent_vectors.detach()}


