import glob
import json
import torch
import shutil
import os
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from typing import Dict
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import AdamW
from evaluate import compute_metrics
from doc import Dataset, Example, collate, eval_collate, ent_collate, load_data
from utils import AverageMeter, ProgressMeter
from utils import save_checkpoint, delete_old_ckt, report_num_trainable_parameters, move_to_cuda, get_model_obj, add_scalars, add_direction_to_metric_log
from metric import accuracy
from models import build_model, BiLinker, SiameseOutput
from dict_hub import build_tokenizer, get_tokenizer
from logger_config import logger
import wandb
import tqdm
import json
from tensorboardX import SummaryWriter
from triplet import EntityDict
from rerank import rerank_by_graph



class Trainer:

    def __init__(self, args):
        self.args = args

        self.tensor_logger = SummaryWriter("./tensorboardLog")
        self.step = 0
        # create model
        logger.info("=> creating model")

        ####### contrastive learning #####
        self.model = BiLinker(self.args).cuda()
        logger.info(self.model)

        # define loss function (criterion) and optimizer
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad],
                               lr=args.lr,
                               weight_decay=args.weight_decay)

        # report_num_trainable_parameters(self.model)
        self._setup_training()
        self.train_dataset = Dataset(path=args.train_path, task=args.task)
        self.valid_dataset = Dataset(path=args.valid_path, task=args.task) if args.valid_path else None
        # TODO remove
        self.test_dataset = Dataset(path=args.test_path, task=args.task) if args.test_path else None
        self.train_dataset = ConcatDataset([self.train_dataset, self.valid_dataset, self.test_dataset])
        num_training_steps = args.epochs * len(self.train_dataset) // max(args.batch_size, 1)
        args.warmup = min(args.warmup, num_training_steps // 10)
        logger.info('Total training steps: {}, warmup steps: {}'.format(num_training_steps, args.warmup))
        self.scheduler = self._create_lr_scheduler(self.optimizer, num_training_steps)
        self.best_metric = None

        logger.info('building entity dictionary for evaluation')
        self.entity_dict = EntityDict(entity_dict_dir=os.path.dirname(args.test_path))
        logger.info('entity dictionary successful')

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True)

        self.valid_loader = None
        if self.valid_dataset:
            self.valid_loader = torch.utils.data.DataLoader(
                self.valid_dataset,
                batch_size=args.batch_size * 2,
                shuffle=True,
                collate_fn=collate,
                num_workers=args.workers,
                pin_memory=True)

    def train_loop(self):
        if self.args.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # self._run_eval(0)

        wandb.watch(self.model)
        for epoch in range(self.args.epochs):
            # train for one epoch
            self.train_epoch(epoch)
            self._run_eval(epoch=epoch)

        self.tensor_logger.close()

    @torch.no_grad()
    def _run_eval(self, epoch, step=0):
        metrics = self.eval(epoch)
        wandb.log(metrics['total'])
        wandb.log(metrics['forward'])
        wandb.log(metrics['backward'])


        logger.info('Epoch {}, forward metric: {}'.format(epoch, json.dumps(metrics['forward'])))
        logger.info('Epoch {}, backward metric: {}'.format(epoch, json.dumps(metrics['backward'])))
        logger.info('Epoch {}, average metric: {}'.format(epoch, json.dumps(metrics['total'])))

        os.makedirs("{}/logs".format("/".join(self.args.eval_model_path.split("/")[:-1])), exist_ok=True)

        with open("{}/logs/metrics.json".format("/".join(self.args.eval_model_path.split("/")[:-1])),'a', encoding="utf-8") as writer:
            writer.write("forward metrics: {}\n".format(json.dumps(metrics['forward'])))
            writer.write("backward metrics: {}\n".format(json.dumps(metrics['backward'])))
            writer.write("metrics: {}\n".format(json.dumps(metrics['total'])))

        is_best = self.valid_loader and (self.best_metric is None or metrics['total']['hit10'] > self.best_metric['hit10'])
        if is_best:
            self.best_metric = metrics['total']

        filename = '{}/checkpoint_{}_{}.mdl'.format(self.args.model_dir, epoch, step)
        if step == 0:
            filename = '{}/checkpoint_epoch{}.mdl'.format(self.args.model_dir, epoch)
        save_checkpoint({
            'epoch': epoch,
            'args': self.args.__dict__,
            'state_dict': self.model.state_dict(),
        }, is_best=is_best, filename=filename)
        delete_old_ckt(path_pattern='{}/checkpoint_*.mdl'.format(self.args.model_dir),
                       keep=self.args.max_to_keep)


    @torch.no_grad()
    def predict_by_entities(self, entity_exs, direction="forward") -> torch.tensor:
        examples = []
        for entity_ex in entity_exs:
            if direction =='forward':
                examples.append(Example(head_id='', relation='',
                                    tail_id=entity_ex.entity_id))
            elif direction =='backward':
                examples.append(Example(head_id=entity_ex.entity_id, relation='',
                                    tail_id=''))
        data_loader = torch.utils.data.DataLoader(
            Dataset(path='', examples=examples, task=self.args.task),
            num_workers=4,
            batch_size=1024,
            collate_fn= ent_collate,
            shuffle=False)

        ent_tensor_list = []
        for idx, batch_dict in enumerate(tqdm.tqdm(data_loader)):
            batch_dict['only_ent_embedding'] = True
            batch_dict = move_to_cuda(batch_dict)
            batch_dict.update({"direction": direction})
            entity_vector = self.model(**batch_dict, hr_batch_dict=None, tr_batch_dict=None)
            ent_tensor_list.append(entity_vector)

        return torch.cat(ent_tensor_list, dim=0)

    @torch.no_grad()
    def eval(self, epoch):
        all_triples = load_data(self.args.test_path, add_forward_triplet=True, add_backward_triplet=False)
        eval_loader = DataLoader(Dataset(path='', examples=all_triples, task=self.args.task),
                                    num_workers=4,
                                    batch_size=1024,
                                    collate_fn=eval_collate,
                                    shuffle=False)

        losses = AverageMeter('Loss', ':.4')

        hr_tensor_list = []
        tr_tensor_list = []

        for i, batch_dict in enumerate(eval_loader):
            self.model.eval()

            if torch.cuda.is_available():
                batch_dict = move_to_cuda(batch_dict)
            batch_size = len(batch_dict['batch_data'])

            outputs = self.model(**batch_dict)  # encode
            outputs = get_model_obj(self.model).compute_logits(output_dict=outputs, batch_dict=batch_dict)  # compare
            outputs = SiameseOutput(**outputs)

            hr_tensor_list.append(outputs.hr_vector)
            tr_tensor_list.append(outputs.tr_vector)


            logit_head, logit_tail, labels = outputs.logit_head, outputs.logit_tail, outputs.labels
            loss_head = self.criterion(logit_head, labels)
            loss_tail = self.criterion(logit_tail, labels)
            loss = 0.5*(loss_head + loss_tail) 

            losses.update(loss.item(), batch_size)


        hr_tensors = torch.cat(hr_tensor_list, 0)
        ent_embs = self.predict_by_entities(self.entity_dict.entity_exs)

        forward_labels = [self.entity_dict.entity_to_idx(triple.tail_id) for triple in all_triples]
        _, _, forward_metrics, _ = compute_metrics(hr_tensor=hr_tensors,
                                                                                entities_tensor=ent_embs,
                                                                                target=forward_labels,
                                                                                examples=all_triples,   # filter out similar tail entities
                                                                                batch_size=256)
        del hr_tensors, ent_embs, hr_tensor_list

        tr_tensors = torch.cat(tr_tensor_list, 0)
        ent_embs = self.predict_by_entities(self.entity_dict.entity_exs, direction="backward")
        backward_labels = [self.entity_dict.entity_to_idx(triple.head_id) for triple in all_triples]
        _, _, backward_metrics, _ = compute_metrics(hr_tensor=tr_tensors,
                                                                                entities_tensor=ent_embs,
                                                                                target=backward_labels,
                                                                                examples=all_triples,
                                                                                batch_size=256)

        del tr_tensors, ent_embs, tr_tensor_list

        metrics = {k: round((forward_metrics[k] + backward_metrics[k]) / 2, 4) for k in forward_metrics}

        forward_metrics = add_direction_to_metric_log(forward_metrics, 'forward')
        wandb.log(forward_metrics)
        wandb.log({"losses":losses.avg})

        add_scalars(self.tensor_logger, forward_metrics, epoch)
        add_scalars(self.tensor_logger, {"losses":losses.avg}, epoch)

        backward_metrics = add_direction_to_metric_log(backward_metrics, 'backward')
        wandb.log(backward_metrics)
        add_scalars(self.tensor_logger, backward_metrics, epoch)

        logger.info('Epoch {}, forward metric: {}'.format(epoch, json.dumps(forward_metrics)))
        logger.info('Epoch {}, backward metric: {}'.format(epoch, json.dumps(backward_metrics)))
        logger.info('Epoch {}, average metric: {}'.format(epoch, json.dumps(metrics)))

        os.makedirs("{}/logs".format("/".join(self.args.eval_model_path.split("/")[:-1])), exist_ok=True)

        with open("{}/logs/metrics.json".format("/".join(self.args.eval_model_path.split("/")[:-1])),'a', encoding="utf-8") as writer:
            writer.write("forward metrics: {}\n".format(json.dumps(forward_metrics)))
            writer.write("backward metrics: {}\n".format(json.dumps(backward_metrics)))
            writer.write("metrics: {}\n".format(json.dumps(metrics)))

        return {"total":metrics, "forward":forward_metrics, "backward":backward_metrics}

    def train_epoch(self, epoch):
        losses = AverageMeter('Loss', ':.4')
        losses_forward = AverageMeter('LossForward', ':.3')
        losses_backward = AverageMeter('LossBackward', ':.3')

        head_top1 = AverageMeter('Head Acc@1', ':6.2f')
        head_top10 = AverageMeter('Head Acc@10', ':6.2f')

        tail_top1 = AverageMeter('Tail Acc@1', ':6.2f')
        tail_top10 = AverageMeter('Tail Acc@10', ':6.2f')
        t = AverageMeter('InvT', ':6.2f')
        hr_tr_scale = AverageMeter('s', ':6.2f')

        progress = ProgressMeter(
            len(self.train_loader),
            [losses,losses_forward, losses_backward, head_top1, tail_top1, t, hr_tr_scale],
            prefix="Epoch: [{}]".format(epoch))


        for i, batch_dict in enumerate(self.train_loader):
            # switch to train mode
            self.model.train()
            if torch.cuda.is_available():
                batch_dict = move_to_cuda(batch_dict)
            batch_size = len(batch_dict['batch_data'])

            # compute output
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch_dict)
            else:
                outputs = self.model(**batch_dict)
            outputs = get_model_obj(self.model).compute_logits(output_dict=outputs, batch_dict=batch_dict)  # compare
            outputs = SiameseOutput(**outputs)

            logit_forward, logit_backward, labels = outputs.logit_head, outputs.logit_tail, outputs.labels
            assert logit_forward.size(0) == batch_size
            # head + relation [MASK] -> tail
            loss_forward = self.criterion(logit_forward, labels)
            # tail -> head + relation + [MASK]（symmetric)
            loss_forward += self.criterion(logit_forward[:, :batch_size].t(), labels)

            # [MASK] + relation tail -> head
            loss_backward = self.criterion(logit_backward, labels)
            # head -> [MASK] + relation tail
            loss_backward += self.criterion(logit_backward[:, :batch_size].t(), labels)

            loss = 0.5 * (loss_forward + loss_backward)
            losses.update(loss.item(), batch_size)

            head_acc1, head_acc10 = accuracy(logit_forward, labels, topk=(1, 10))
            tail_acc1, tail_acc10 = accuracy(logit_backward, labels, topk=(1, 10))

            head_top1.update(head_acc1.item(), batch_size)
            head_top10.update(head_acc10.item(), batch_size)
            tail_top1.update(tail_acc1.item(), batch_size)
            tail_top10.update(tail_acc10.item(), batch_size)

            losses_forward.update(loss_forward.item(), batch_size)
            losses_backward.update(loss_backward.item(), batch_size)

            t.update(outputs.t, 1)
            hr_tr_scale.update(outputs.hr_tr_s, 1)
            losses.update(loss.item(), batch_size)

            # log every step
            if self.step % self.args.eval_every_n_step == 0:
                wandb.log({"losses": losses.avg, "head_acc1": head_top1.avg, "head_acc10": head_top10.avg, "tail_acc1": tail_top1.avg, "tail_acc10": tail_top10.avg, "t": t.avg})
                wandb.log({"lr": self.scheduler.get_last_lr()[0]})
            add_scalars(self.tensor_logger, {"losses": losses.avg,
                                                                "head_acc1": head_top1.avg, "head_acc10": head_top10.avg, 
                                                                "tail_acc1": tail_top1.avg, "tail_acc10": tail_top10.avg, "t": t.avg,
                                                                "lr": self.scheduler.get_last_lr()[0]
                                                                },
                                           self.step)

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
            self.step += 1

            if i % self.args.print_freq == 0:
                progress.display(i)
            if self.step % self.args.eval_every_n_step == 0:
                self._run_eval(epoch=epoch, step=i + 1)

        logger.info('Learning rate: {}'.format(self.scheduler.get_last_lr()[0]))

    def _setup_training(self):
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model).cuda()
        elif torch.cuda.is_available():
            self.model.cuda()
        else:
            logger.info('No gpu will be used')

    def _create_lr_scheduler(self, optimizer, num_training_steps):
        if self.args.lr_scheduler == 'linear':
            return get_linear_schedule_with_warmup(optimizer=optimizer,
                                                   num_warmup_steps=self.args.warmup,
                                                   num_training_steps=num_training_steps)
        elif self.args.lr_scheduler == 'cosine':
            return get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                   num_warmup_steps=self.args.warmup,
                                                   num_training_steps=num_training_steps)
        else:
            assert False, 'Unknown lr scheduler: {}'.format(self.args.scheduler)
