import torch.nn as nn
import torch.utils.data
import json
import torch
import os
import shutil
import glob
from typing import Dict
from transformers import AdamW, get_linear_schedule_with_warmup
from preprocess_data import collate
from utils import AverageMeter, ProgressMeter, logger
from metric import accuracy
from transformers import AutoModel, AutoConfig
from dataclasses import dataclass, field

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
    train_batch_size: int = field(default=128,
                        metadata={"help": "train batch size"})
    eval_batch_size: int = field(default=128,
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
                 num_workers=4,
                 train_args: TrainingArguments = None
                 ):
        # training arguments
        self.args = train_args
        self.pretrained_model_path = pretrained_model_path
        self.eval_model_path = eval_model_path
        # create model
        logger.info("Creating model")
        self.config = AutoConfig.from_pretrained(pretrained_model_path)
        self.model = AutoModel.from_pretrained(pretrained_model_path)
        # adding mention span token type ids
        old_type_vocab_size = self.config.type_vocab_size
        self.config.type_vocab_size = 3
        new_token_type_embeddings = nn.Embedding(self.config.type_vocab_size, self.config.hidden_size)
        self.model._init_weights(new_token_type_embeddings)
        new_token_type_embeddings.weight.data[:old_type_vocab_size, :] = self.model.embeddings.token_type_embeddings[:old_type_vocab_size, :]
        self.model.embeddings.token_type_embeddings = new_token_type_embeddings
        self.model.predict_head = nn.Linear(self.config.hidden_size, 1)
        self.t = 20
        self._setup_training()

        # loss and optimization
        self.criterion = nn.BCEWithLogitsLoss().cuda()
        self.optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad],
                               lr=self.args.learning_rate,
                               weight_decay=self.args.weight_decay)

        num_training_steps = args.epochs * len(train_dataset) // max(self.args.train_batch_size, 1)
        self.args.warmup = min(args.warmup, num_training_steps // 10)
        logger.info('Total training steps: {}, warmup steps: {}'.format(num_training_steps, self.args.warmup))
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                   num_warmup_steps=self.args.warmup,
                                                   num_training_steps=num_training_steps)

        # initial status
        self.is_training = True
        self.best_metric = None
        self.epoch = 0
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

    def run(self):

        for epoch in range(self.args.epochs):
            self.train_one_epoch()
            self.evaluate()
            self.epoch = epoch

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
            metric_dict = self.eval_loop()
        else:
            metric_dict = self.eval_loop()
            if self.best_metric is None or metric_dict['Acc@1'] > self.best_metric['Acc@1']:
                self.best_metric = metric_dict
                with open(os.path.join(self.eval_model_path, "best_metric"), 'w', encoding='utf-8') as f:
                    f.write(json.dumps(metric_dict, indent=4))

                self.save_checkpoint({
                    'epoch': self.epoch,
                    'args': self.args.__dict__,
                    'state_dict': self.model.state_dict(),
                }, is_best=True, filename="best_model.ckpt")

            else:
                filename = '{}/checkpoint_{}_{}.ckpt'.format(self.eval_model_path, self.epoch, step)
                self.save_checkpoint({
                    'epoch': self.epoch,
                    'args': self.args.__dict__,
                    'state_dict': self.model.state_dict(),
                    }, is_best=False, filename=filename)

        logger.info(metric_dict)
        self.delete_old_ckt(path_pattern='{}/checkpoint_*.ckpt'.format(self.eval_model_path),
                       keep=self.args.max_weights_to_keep)

    @torch.no_grad()
    def eval_loop(self) -> Dict:

        losses = AverageMeter('Loss', ':.4')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top3 = AverageMeter('Acc@3', ':6.2f')
        top10 = AverageMeter('Acc@10', ':6.2f')

        for i, batch_dict in enumerate(self.valid_loader):
            self.model.eval()

            if torch.cuda.is_available():
                batch_dict = self.move_to_cuda(batch_dict)
            batch_size = len(batch_dict['batch_data'])

            outputs = self.model(**batch_dict)
            h = outputs.last_hidden_state
            logits = self.model.predict_head(h) * self.t
            labels = batch_dict['labels']
            loss = self.criterion(logits, labels)
            losses.update(loss.item(), batch_size)

            acc1, acc3, acc10 = accuracy(logits, labels, topk=(1, 3, 10))
            top1.update(acc1.item(), batch_size)
            top3.update(acc3.item(), batch_size)
            top10.update(acc10.item(), batch_size)

        metric_dict = {'Acc@1': round(top1.avg, 3),
                       'Acc@3': round(top3.avg, 3),
                       'Acc@10': round(top10.avg, 3),
                       'loss': round(losses.avg, 3)}
        logger.info('Epoch {}, valid metric: {}'.format(self.epoch, json.dumps(metric_dict)))
        return metric_dict

    def train_one_epoch(self):
        losses = AverageMeter('Loss', ':.4')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top3 = AverageMeter('Acc@3', ':6.2f')
        progress = ProgressMeter(
            len(self.train_loader),
            [losses, top1, top3],
            prefix="Epoch: [{}]".format(self.epoch))

        for i, batch_dict in enumerate(self.train_loader):
            self.model.train()

            if torch.cuda.is_available():
                batch_dict = self.move_to_cuda(batch_dict)
            batch_size = len(batch_dict['batch_data'])
            labels = batch_dict['labels']


            outputs = self.model(**batch_dict)
            h = outputs.last_hidden_state
            logits = self.model.predict_head(h) * self.t
            loss = self.criterion(logits, labels)

            acc1, acc3 = accuracy(logits, labels, topk=(1, 3))
            top1.update(acc1.item(), batch_size)
            top3.update(acc3.item(), batch_size)

            losses.update(loss.item(), batch_size)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.optimizer.step()
            self.scheduler.step()

            if i % self.args.log_every_n_steps == 0:
                progress.display(i)
            if (i + 1) % self.args.eval_every_n_steps == 0:
                self.eval_loop(epoch=self.epoch, step=i + 1)

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

