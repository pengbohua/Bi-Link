from cProfile import label
import torch.nn as nn
import torch.utils.data
import time
import json
import torch
import os
import shutil
import glob
from typing import Dict, List
from transformers import AdamW, get_linear_schedule_with_warmup
from preprocess_data import collate
from utils import AverageMeter, ProgressMeter, logger, compute_mlm_batch_scores
from transformers import BertModel, AutoConfig
from dataclasses import dataclass, field
from metrics import compute_metric
from tqdm import tqdm

curr_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))

@dataclass
class TrainingArguments:
    learning_rate: float = field(default=1e-5,
                            metadata={"help": "learning rate for optimization"}
                            )
    weight_decay: float = field(default=1e-4,
                            metadata={"help": "weight decay parameter for optimization"}
                            )
    grad_clip: float = field(default=10,
                            metadata={"help": "magnitude for gradient clipping"}
                            )
    epochs: int = field(default=3,
                            metadata={"help": "number of training epochs"}
                            )
    warmup: int = field(default=500,
                        metadata={"help": "warmup steps"})
    train_batch_size: int = field(default=2,
                        metadata={"help": "train batch size"})
    eval_batch_size: int = field(default=2,
                        metadata={"help": "eval batch size"})
    eval_every_n_steps: int = field(default=1000,
                        metadata={"help": "eval every n steps"})
    log_every_n_steps: int = field(default=100,
                        metadata={"help": "log every n steps"})
    max_weights_to_keep: int = field(default=3,
                                     metadata={"help": "max number of weight file to keep"})

class Trainer:

    def __init__(self,
                 pretrained_model_path,
                 eval_model_path,
                 train_dataset,
                 eval_dataset,
                 num_workers=1,
                 train_args: TrainingArguments = None
                 ):
        # training arguments
        self.args = train_args
        self.pretrained_model_path = pretrained_model_path
        self.eval_model_path = eval_model_path + "/" + curr_time
        os.makedirs(self.eval_model_path, exist_ok=True)
        # create model
        logger.info("Creating model")
        self.config = AutoConfig.from_pretrained(pretrained_model_path)
        self.model = BertModel.from_pretrained(pretrained_model_path)
        # adding mention span as a new type to token type ids
        old_type_vocab_size = self.config.type_vocab_size
        self.config.type_vocab_size = 3
        new_token_type_embeddings = nn.Embedding(self.config.type_vocab_size, self.config.hidden_size)
        self.model._init_weights(new_token_type_embeddings)
        new_token_type_embeddings.weight.data[:old_type_vocab_size, :] = self.model.embeddings.token_type_embeddings.weight.data[:old_type_vocab_size, :]
        self.model.embeddings.token_type_embeddings = new_token_type_embeddings
        self.model.classification_head = nn.Sequential(
                                    nn.Linear(self.config.hidden_size, 2)
                                )
        self.model.b_embedding = nn.Embedding(self.config.vocab_size, self.config.hidden_size, sparse=False)
        self.model.b_embedding.weight.data = self.model.embeddings.word_embeddings.weight.data
        print(self.model.b_embedding.weight.data.shape)
        print(self.model.embeddings.word_embeddings.weight.data.shape)
        # self.t = 20
        self._setup_training()
        
        # loss and optimization
        self.criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.tokenizer.pad_token_id).cuda()
        self.optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad],
                               lr=self.args.learning_rate,
                               weight_decay=self.args.weight_decay)

        num_training_steps = self.args.epochs * len(train_dataset) // max(self.args.train_batch_size, 1)
        self.args.warmup = min(self.args.warmup, num_training_steps // 10)
        logger.info('Total training steps: {}, warmup steps: {}'.format(num_training_steps, self.args.warmup))
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                   num_warmup_steps=self.args.warmup,
                                                   num_training_steps=num_training_steps)

        # initial status
        self.is_training = True
        self.best_metric = None
        self.epoch = 0
        self.mask_token_id = train_dataset.tokenizer.mask_token_id
        self.pad_token_id = train_dataset.tokenizer.pad_token_id
        # dataloader
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True)

        self.valid_loader = torch.utils.data.DataLoader(
                eval_dataset,
                batch_size=self.args.eval_batch_size,
                shuffle=True,
                collate_fn=collate,
                num_workers=num_workers,
                pin_memory=True)
        
        self.valid_loader = torch.utils.data.DataLoader(
                eval_dataset,
                batch_size=self.args.eval_batch_size,
                shuffle=True,
                collate_fn=collate,
                num_workers=num_workers,
                pin_memory=True)

    def run(self):

        for epoch in range(self.args.epochs):
            self.epoch = epoch
            
            self.train_one_epoch()
            if epoch >= 2:
                self.evaluate()
        print("Training finished!")

    @staticmethod
    def move_to_cuda(sample):
        if len(sample) == 0:
            return {}

        def _move_to_cuda(maybe_tensor):
            if torch.is_tensor(maybe_tensor):
                return maybe_tensor.cuda(non_blocking=True)
            else:
                for key, value in maybe_tensor.items():
                    maybe_tensor[key] = _move_to_cuda(value)
                return maybe_tensor
        return _move_to_cuda(sample)

    @torch.no_grad()
    def evaluate(self, step=0):
        if not self.is_training:
            metric_dict = self.mlm_eval_loop()
        else:
            metric_dict = self.mlm_eval_loop()
            if self.best_metric is None or metric_dict['hit1'] > self.best_metric['hit1']:
                self.best_metric = metric_dict
                with open(os.path.join(self.eval_model_path, "best_metric"), 'w', encoding='utf-8') as f:
                    f.write(json.dumps(metric_dict, indent=4))
                self.save_checkpoint({
                    'epoch': self.epoch,
                    'state_dict': self.model.state_dict(),
                }, is_best=True, filename="best_model.ckpt")

            else:
                filename = '{}/checkpoint_{}_{}.ckpt'.format(self.eval_model_path, self.epoch, step)
                self.save_checkpoint({
                    'epoch': self.epoch,
                    'state_dict': self.model.state_dict(),
                    }, is_best=False, filename=filename)

        logger.info(metric_dict)
        self.delete_old_ckt(path_pattern='{}/checkpoint_*.ckpt'.format(self.eval_model_path),
                       keep=self.args.max_weights_to_keep)
    
    @torch.no_grad()
    def mlm_eval_loop(self) -> Dict:

        losses = AverageMeter('Loss', ':.4')
        accs = AverageMeter('Acc', ':6.2f')
        hit1 = AverageMeter('hit1', ':6.2f')
        hit3 = AverageMeter('hit3', ':6.2f')
        hit10 = AverageMeter('hit10', ':6.2f')
        mrr = AverageMeter('MRR', ':6.2f')
        
        for i, (batch_dict, raw_input_ids, labels) in tqdm(enumerate(self.valid_loader)):
            self.model.eval()
            if i > 100:
                break
            if torch.cuda.is_available():
                batch_dict = self.move_to_cuda(batch_dict)
                raw_input_ids = self.move_to_cuda(raw_input_ids)
                labels = self.move_to_cuda(labels)
            batch_size = len(batch_dict['input_ids'])

            outputs = self.model(**batch_dict)
            input_ids = batch_dict["input_ids"]

            # do not use mask for prediction
            batch_dict["input_ids"] = raw_input_ids
            
            outputs = self.model(**batch_dict)
            logits = self.get_model_obj(self.model).mlm_head(outputs.last_hidden_state)
            mt_labels = raw_input_ids[input_ids == 103]
            loss = self.criterion(logits[input_ids == 103].squeeze(), mt_labels.long())

            batch_scores = compute_mlm_batch_scores(logits, input_ids, raw_input_ids, 103)
            metrics = compute_metric(batch_scores, labels)
            mrr.update(metrics['mrr'], metrics['chunk_size'])
            hit1.update(metrics['hit1'], metrics['chunk_size'])
            hit3.update(metrics['hit3'], metrics['chunk_size'])
            hit10.update(metrics['hit10'], metrics['chunk_size'])
            losses.update(loss.item(), batch_size)

        metric_dict = {'acc': round(accs.avg, 3),
                       'mrr': round(mrr.avg, 3),
                       'hit1': round(hit1.avg, 3),
                       'hit3': round(hit3.avg, 3),
                       'hit10': round(hit10.avg, 3),
                       'loss': round(losses.avg, 3)}
        logger.info('Epoch {}, valid metric: {}'.format(self.epoch, json.dumps(metric_dict)))
        return metric_dict

    def train_one_epoch(self):
        losses = AverageMeter('Loss', ':.4')
        top1 = AverageMeter('Acc@1', ':6.2f')
        # top3 = AverageMeter('Acc@3', ':6.2f')
        progress = ProgressMeter(
            len(self.train_loader),
            [losses, top1],
            prefix="Epoch: [{}]".format(self.epoch))

        for i, (batch_dict, b_input_ids, labels) in enumerate(self.train_loader):
            self.model.train()

            if torch.cuda.is_available():
                batch_dict = self.move_to_cuda(batch_dict)
                b_input_ids = self.move_to_cuda(b_input_ids)
                labels = self.move_to_cuda(labels)
            input_ids = batch_dict["input_ids"]
            batch_size = len(input_ids)

            outputs = self.model(**batch_dict)
            mention_logits = outputs.last_hidden_state[input_ids == self.mask_token_id]
            b_h = self.get_model_obj(self.model).b_embedding(b_input_ids)
            inner_product = torch.mul(mention_logits.unsqueeze(dim=1), b_h)
            inner_product = self.get_model_obj(self.model).classification_head(inner_product)
            inner_product = torch.softmax(inner_product, dim=2)
            # print(inner_product.shape)
            inner_product = torch.mean(inner_product, dim=1)
            # print(inner_product.shape)
            loss = self.criterion(inner_product.squeeze(), labels.long())
            acc = inner_product.max(dim=1)[1].eq(labels.squeeze()).sum()
            # only update mention loss and acc
            top1.update(acc.item(), batch_size)
            losses.update(loss.item(), batch_size)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.optimizer.step()
            self.scheduler.step()

            if i % self.args.log_every_n_steps == 0:
                progress.display(i)
            if (i + 1) % self.args.eval_every_n_steps == 0:
                self.mlm_eval_loop()

    @staticmethod
    def accuracy(output: torch.tensor, target: torch.tensor, topk=(1,)) -> List[torch.tensor]:
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()

            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    @staticmethod
    def get_model_obj(model: nn.Module):
        return model.module if hasattr(model, "module") else model

    @staticmethod
    def save_checkpoint(state: dict, is_best: bool, filename: str):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.dirname(filename) + '/model_best.ckpt')
        shutil.copyfile(filename, os.path.dirname(filename) + '/model_last.ckpt')

    @staticmethod
    def delete_old_ckt(path_pattern: str, keep=5):
        files = sorted(glob.glob(path_pattern), key=os.path.getmtime, reverse=True)     # glob search cur dir with path_pattern
        for f in files[keep:]:
            logger.info('Delete old checkpoint {}'.format(f))
            os.system('rm -f {}'.format(f))

    def _setup_training(self):
        if torch.cuda.device_count() > 1:
            logger.info("Training with {} GPUs in parallel".format(torch.cuda.device_count()))
            self.model = torch.nn.DataParallel(self.model).cuda()
        elif torch.cuda.is_available():
            self.model.cuda()
        else:
            logger.info("Training with CPU")
