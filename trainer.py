import copy
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
from cl_preprocess_data import compose_collate
from utils import AverageMeter, ProgressMeter, logger
from transformers import BertModel, AutoConfig
from dataclasses import dataclass, field
from metrics import accuracy, compute_metric
from models import EntityLinker


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
    num_cand: int = field(default=64,
                            metadata={"help": "number of negative samples"}
                            )
    warmup: int = field(default=500,
                        metadata={"help": "warmup steps"})
    use_amp: bool = field(default=True,
                        metadata={"help": "use mixed precision"})
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
    cut_off_negative_gradients: bool = field(default=bool,
                                             metadata={"help": "cut off gradient flow to negative samples"})

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
        self.num_candidates = self.args.num_cand
        self.pretrained_model_path = pretrained_model_path
        self.eval_model_path = eval_model_path + "/" + curr_time
        os.makedirs(self.eval_model_path, exist_ok=True)
        # create model
        logger.info("Creating model")
        self.config = AutoConfig.from_pretrained(pretrained_model_path)
        self.model = EntityLinker(pretrained_model_path)
        self._setup_training()

        self.criterion = nn.CrossEntropyLoss().cuda()
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
        # dataloader
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            collate_fn=compose_collate,
            num_workers=1,
            pin_memory=True,
            drop_last=True)

        self.valid_loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            collate_fn=compose_collate,
            num_workers=1,
            pin_memory=True)

    def run(self):
        if self.args.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.epochs):
            self.train_one_epoch()
            self.evaluate()
            self.epoch += 1

    @staticmethod
    def move_to_cuda(sample):
        if len(sample) == 0:
            return {}

        def _move_to_cuda(maybe_tensor):
            if torch.is_tensor(maybe_tensor):
                return maybe_tensor.cuda(non_blocking=True)
            elif type(maybe_tensor) == list:
                return [_move_to_cuda(t) for t in maybe_tensor]
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
            if self.best_metric is None or metric_dict['hit1'] > self.best_metric['hit1']:
                self.best_metric = metric_dict
                with open(os.path.join(self.eval_model_path, "best_metric"), 'w', encoding='utf-8') as f:
                    f.write(json.dumps(metric_dict, indent=4))

                self.save_checkpoint(self.model.state_dict(),
                                     is_best=True, filename=os.path.join(self.eval_model_path, "best_model.ckpt"))

            else:
                filename = '{}/checkpoint_{}_{}.ckpt'.format(self.eval_model_path, self.epoch, step)
                self.save_checkpoint(self.model.state_dict(), is_best=False, filename=filename)

            self.delete_old_ckt(path_pattern='{}/checkpoint_*.ckpt'.format(self.eval_model_path),
                       keep=self.args.max_weights_to_keep)

        logger.info(metric_dict)

    @torch.no_grad()
    def eval_loop(self) -> Dict:

        losses = AverageMeter('Loss', ':.4')
        accs = AverageMeter('Acc', ':6.2f')
        hit1 = AverageMeter('hit1', ':6.2f')
        hit3 = AverageMeter('hit3', ':6.2f')
        hit10 = AverageMeter('hit10', ':6.2f')
        mrr = AverageMeter('MRR', ':6.2f')

        for i, batch_cl_data in enumerate(self.valid_loader):
            self.model.eval()

            if torch.cuda.is_available():
                batch_cl_data = self.move_to_cuda(batch_cl_data)

            mention_dicts = batch_cl_data["mention_dicts"]
            candidate_dicts = batch_cl_data["candidate_dicts"]
            labels = batch_cl_data["labels"]

            batch_size = len(labels)

            logits, metrics = self.get_model_obj(self.model).predict(mention_dicts, candidate_dicts, labels)
            loss = self.criterion(logits, labels)
            losses.update(loss.item(), batch_size)

            predictions = logits.argmax(1)
            _acc = torch.sum(torch.eq(predictions, labels)) / len(labels)

            accs.update(_acc, batch_size)
            mrr.update(metrics['mrr'], batch_size)
            hit1.update(metrics['hit1'], batch_size)
            hit3.update(metrics['hit3'], batch_size)
            hit10.update(metrics['hit10'], batch_size)

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
        accs = AverageMeter('Acc', ':6.2f')
        progress = ProgressMeter(
            len(self.train_loader),
            [losses, accs],
            prefix="Epoch: [{}]".format(self.epoch))

        for i, batch_cl_data in enumerate(self.train_loader):
            # switch to train mode
            self.model.train()

            if torch.cuda.is_available():
                batch_cl_data = self.move_to_cuda(batch_cl_data)

            mention_dicts = batch_cl_data["mention_dicts"]
            entity_dicts = batch_cl_data["entity_dicts"]
            candidate_dicts = batch_cl_data["candidate_dicts"]
            batch_size = len(mention_dicts)

            # compute output

            output_dicts = self.model(mention_dicts=mention_dicts, entity_dicts=entity_dicts, candidate_dict_list=candidate_dicts)
            logits = self.get_model_obj(self.model).compute_logits(**output_dicts)
            labels = torch.arange(len(logits)).to(logits.device)

            predictions = logits.argmax(1)
            print("predictions", predictions)
            _acc = torch.sum(torch.eq(predictions, labels)) / len(labels)

            loss = self.criterion(logits, labels)

            accs.update(_acc.item(), batch_size)
            losses.update(loss.item(), batch_size)

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            if self.args.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.optimizer.step()
            self.scheduler.step()

            if i % self.args.log_every_n_steps == 0:
                progress.display(i)
            if (i + 1) % self.args.eval_every_n_steps == 0:
                self.evaluate(step=i)

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

