import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from typing import Dict
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import AdamW

from doc_rel import Dataset, collate, load_data, Example
from evaluate import compute_metrics
from utils import AverageMeter, ProgressMeter, add_scalars
from utils import save_checkpoint, delete_old_ckt, report_num_trainable_parameters, move_to_cuda, get_model_obj, add_scalars, add_direction_to_metric_log
from metric import accuracy
from model_prefix import ModelOutput, CustomBertModel
from dict_hub import build_tokenizer, get_tokenizer
from logger_config import logger
import itertools
import tqdm
from torch.utils.tensorboard import SummaryWriter
import json
from triplet import EntityDict


class Trainer:

    def __init__(self, args, ngpus_per_node):
        self.args = args
        self.temp_path = self.args.template_path
        self.ngpus_per_node = ngpus_per_node
        tokenizer = get_tokenizer()
        self.tensor_logger = SummaryWriter("./tensorboardLog")
        self.step = 0
        # create model
        logger.info("=> creating model")

        ####### contrastive learning #####
        self.args.use_amp = False
        self.model = CustomBertModel(self.args,).cuda()
        logger.info(self.model)

        # define loss function (criterion) and optimizer
        self.criterion = nn.CrossEntropyLoss().cuda()

        self.optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad],
                               lr=args.lr,
                               weight_decay=args.weight_decay)

        # report_num_trainable_parameters(self.model)
        self._setup_training()
        train_temp_path = [os.path.join(self.temp_path, p)for p in ['FB15k237_train_ultra_fine_templates.pickle', 'FB15k237_train_fine_templates.pickle', 'FB15k237_train_coarse_templates.pickle']]
        print("template path", (train_temp_path+[None]))
        self.train_datasets = [Dataset(path=args.train_path, task=args.task, template_path=[temp_path]) for temp_path in (train_temp_path + [None])]
        # self.val_temp_path = os.path.join(self.temp_path, 'FB15k237_test_ultra_fine_templates.pickle')
        self.val_temp_path = None
        self.valid_dataset = Dataset(path=args.valid_path, task=args.task, template_path=[self.val_temp_path]) if args.valid_path else None

        num_training_steps = args.epochs * len(self.train_datasets) * len(self.train_datasets[0]) // max(args.batch_size, 1)
        args.warmup = min(args.warmup, num_training_steps // 10)
        logger.info('Total training steps: {}, warmup steps: {}'.format(num_training_steps, args.warmup))
        self.scheduler = self._create_lr_scheduler(num_training_steps)
        self.best_metric = None

        logger.info('building entity dictionary for evaluation')
        self.entity_dict = EntityDict(entity_dict_dir=os.path.dirname(args.valid_path))
        logger.info('entity dictionary successful')

        max_dst_len = -1
        max_train_dst = None
        for dst in self.train_datasets:
            if len(dst) > max_dst_len:
                max_dst_len = len(dst)
                max_train_dst = dst

        self.cycle_train_datasets = list(filter(lambda x: len(x) < max_dst_len, self.train_datasets))
        self.cycle_train_dataloaders = [iter(itertools.cycle(DataLoader(
            train_dst,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=args.workers,
            pin_memory=True))) for train_dst in self.cycle_train_datasets]
        
        self.train_loaders = self.cycle_train_dataloaders + [iter(DataLoader(
            max_train_dst,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=args.workers,
            pin_memory=True))]

        self.valid_loader = DataLoader(
                self.valid_dataset,
                batch_size=args.batch_size * 2,
                shuffle=True,
                collate_fn=collate,
                num_workers=args.workers,
                pin_memory=True)
    def reload(self):
        self.cycle_train_dataloaders = [iter(itertools.cycle(DataLoader(
            train_dst,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=self.args.workers,
            pin_memory=True))) for train_dst in self.cycle_train_datasets]
        
        self.train_loaders = self.cycle_train_dataloaders + [iter(DataLoader(
            self.max_train_dst,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=self.args.workers,
            pin_memory=True))]
        
    def train_loop(self):
        if self.args.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        # self._run_eval(0)

        # wandb.watch(self.model, log_freq=1000)
        for epoch in range(self.args.epochs):
            losses = AverageMeter('Loss', ':.4')
            top1 = AverageMeter('Acc@1', ':6.2f')
            top3 = AverageMeter('Acc@3', ':6.2f')
            inv_t = AverageMeter('InvT', ':6.2f')

            # train for one epoch
            progress = ProgressMeter(
            len(self.train_loaders[0])*len(self.train_loaders),
            [losses, inv_t, top1, top3],
            prefix="Epoch: [{}]".format(epoch))
            
            for i in range(len(self.train_loaders[0])):
                for dataloader in self.train_loaders:
                    try:
                        batch_dict = next(dataloader)
                    except StopIteration:
                        break
                    batch_size = len(batch_dict['batch_data'])
                    acc1, acc3, outputs, loss = self.train_one_step(epoch, batch_dict)
                
                    top1.update(acc1.item(), batch_size)
                    top3.update(acc3.item(), batch_size)

                    inv_t.update(outputs.inv_t, 1)
                    losses.update(loss, batch_size)
                    
                if i % self.args.print_freq == 0:
                    progress.display(i)
                if (i + 1) % self.args.eval_every_n_step == 0:
                    self._run_eval(epoch=epoch, step=i + 1)
                self.reload()   # reload and start a new epoch

                # logging
                train_log_dict = {"train_losses": losses.avg,
                              "top1": top1.avg,
                             "top3": top3.avg, "inv_t": inv_t.avg,
                               "lr": self.scheduler.get_last_lr()[0]}
                # wandb.log(train_log_dict)
                add_scalars(self.tensor_logger, train_log_dict, self.step)
            
            self._run_eval(epoch=epoch)


    @torch.no_grad()
    def _run_eval(self, epoch, step=0):
        forward_metrics = self.eval_single_direction(epoch, eval_forward=True)
        backward_metrics = self.eval_single_direction(epoch, eval_forward=False)

        metric_dict = {k: round((forward_metrics[k] + backward_metrics[k]) / 2, 4) for k in forward_metrics}

        logger.info('Epoch {}, forward metric: {}'.format(epoch, json.dumps(forward_metrics)))
        logger.info('Epoch {}, backward metric: {}'.format(epoch, json.dumps(backward_metrics)))
        logger.info('Epoch {}, average metric: {}'.format(epoch, json.dumps(metric_dict)))

        os.makedirs("logs/{}".format(self.args.curr_time), exist_ok=True)

        with open("logs/{}/metric_epoch{}.txt".format(self.args.curr_time, epoch),'a', encoding="utf-8") as writer:
            writer.write("forward metrics: {}\n".format(json.dumps(forward_metrics)))
            writer.write("backward metrics: {}\n".format(json.dumps(backward_metrics)))
            writer.write("metrics: {}\n".format(json.dumps(metric_dict)))

        is_best = self.best_metric is None or metric_dict['hit1'] > self.best_metric['hit1']
        if is_best:
            self.best_metric = metric_dict
            with open(os.path.join(self.args.eval_model_path, "best_metric"), 'w', encoding='utf-8') as f:
                f.write(json.dumps(metric_dict, indent=4))

            save_checkpoint(self.model.state_dict(),
                                 is_best=True, filename=os.path.join(self.args.model_dir, "best_model.ckpt"))

        else:
            filename = '{}/checkpoint_{}_{}.ckpt'.format(self.args.model_dir, epoch, step)
            save_checkpoint(self.model.state_dict(), is_best=False, filename=filename)

        delete_old_ckt(path_pattern='{}/checkpoint_*.ckpt'.format(self.args.eval_model_path),
                            keep=self.args.max_to_keep)
        
    @torch.no_grad()
    def predict_by_entities(self, entity_exs) -> torch.tensor:
        # collect ent embs with tail bert
        examples = []
        for entity_ex in entity_exs:
            examples.append(Example(head_id='', relation='',
                                    tail_id=entity_ex.entity_id))   # tails go in
        data_loader = torch.utils.data.DataLoader(
            Dataset(path='', examples=examples, task=self.args.task),
            num_workers=4,
            batch_size=1024,
            collate_fn=eval_collate,
            shuffle=False)

        ent_tensor_list = []
        for idx, batch_dict in enumerate(tqdm.tqdm(data_loader)):
            batch_dict['only_ent_embedding'] = True
            batch_dict = move_to_cuda(batch_dict)
            outputs = self.model(**batch_dict)
            ent_tensor_list.append(outputs['ent_vectors'])

        return torch.cat(ent_tensor_list, dim=0)


    @torch.no_grad()
    def pretraining_eval_epoch(self, epoch) -> Dict:
        if not self.pre_valid_loader:
            return {}

        losses = AverageMeter('Loss', ':.4')

        for i, batch_dict in enumerate(self.pre_valid_loader):
            self.backbone.eval()

            if torch.cuda.is_available():
                batch_dict = move_to_cuda(batch_dict)
            batch_size = len(batch_dict['labels'])

            outputs = self.backbone(**batch_dict)
            loss = outputs.loss
            losses.update(loss.item(), batch_size)

        torch.save(self.backbone.state_dict(), os.path.join(self.args.model_dir, "bert_uncased_backbone{}".format(epoch)))

        metric_dict = {
                       'PreTraining loss': round(losses.avg, 3)}
        logger.info('Epoch {}, valid metric: {}'.format(epoch, json.dumps(metric_dict)))
        # wandb.log(metric_dict)
        return metric_dict

    @torch.no_grad()
    def eval_single_direction(self, epoch, eval_forward=True,):
        all_triples = load_data(self.args.test_path, self.val_temp_path, add_forward_triplet=eval_forward, add_backward_triplet=not eval_forward)
        eval_loader = DataLoader(Dataset(path='', examples=all_triples, task=self.args.task),
                                    num_workers=4,
                                    batch_size=1024,
                                    collate_fn=eval_collate,
                                    shuffle=False)

        losses = AverageMeter('Loss', ':.4')

        hr_tensor_list = []

        for i, batch_dict in enumerate(eval_loader):
            self.model.eval()

            if torch.cuda.is_available():
                batch_dict = move_to_cuda(batch_dict)
            batch_size = len(batch_dict['batch_data'])

            outputs = self.model(**batch_dict)  # encode
            outputs = get_model_obj(self.model).compute_logits(output_dict=outputs, batch_dict=batch_dict)  # compare
            outputs = ModelOutput(**outputs)


            logits, labels = outputs.logits, outputs.labels
            loss = self.criterion(logits, labels)
            if self.args.do_mlm:
                loss += self.args.mlm_weight*outputs.tail_mlm_loss.mean()
                loss += self.args.mlm_weight*outputs.head_mlm_loss.mean()

            hr_tensor_list.append(outputs.hr_vector)
            losses.update(loss.item(), batch_size)

        hr_tensors = torch.cat(hr_tensor_list, 0)
        ent_embs = self.predict_by_entities(self.entity_dict.entity_exs)

        forward_labels = [self.entity_dict.entity_to_idx(tr.tail_id) for tr in all_triples]
        _, _, metrics, _ = compute_metrics(hr_tensor=hr_tensors,
                                                                                entities_tensor=ent_embs,
                                                                                target=forward_labels,
                                                                                examples=all_triples,   # filter out similar tail entities
                                                                                batch_size=256)
        del hr_tensors, ent_embs, hr_tensor_list

        # logging
        # wandb.log(metrics)
        # add_scalars(self.tensor_logger, metrics, epoch)

        # if eval_forward:
        #     # wandb.log({"losses": losses.avg})
        #     add_scalars(self.tensor_logger, {"losses": losses.avg}, epoch)
        #     add_scalars(self.tensor_logger, add_direction_to_metric_log(metrics, 'forward'), epoch)
        # else:
        #     add_scalars(self.tensor_logger, add_direction_to_metric_log(metrics, 'backward'), epoch)
        return metrics


    @torch.no_grad()
    def eval_epoch(self, epoch) -> Dict:
        if not self.valid_loader:
            return {}

        losses = AverageMeter('Loss', ':.4')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top3 = AverageMeter('Acc@3', ':6.2f')

        for i, batch_dict in enumerate(self.valid_loader):
            self.model.eval()

            if torch.cuda.is_available():
                batch_dict = move_to_cuda(batch_dict)
            batch_size = len(batch_dict['batch_data'])

            outputs = self.model(**batch_dict)  # encode
            outputs = get_model_obj(self.model).compute_logits(output_dict=outputs, batch_dict=batch_dict)  # compare
            outputs = ModelOutput(**outputs)    # output

            logits, labels = outputs.logits, outputs.labels
            loss = self.criterion(logits, labels)
            losses.update(loss.item(), batch_size)

            acc1, acc3 = accuracy(logits, labels, topk=(1, 3))
            top1.update(acc1.item(), batch_size)
            top3.update(acc3.item(), batch_size)

        metric_dict = {'Acc@1': round(top1.avg, 3),
                       'Acc@3': round(top3.avg, 3),
                       'loss': round(losses.avg, 3)}
        logger.info('Epoch {}, valid metric: {}'.format(epoch, json.dumps(metric_dict)))
        # wandb.log(metric_dict)
        # self.tensor_logger.add_scalars("evaluating", metric_dict, self.step)
        return metric_dict

    def mlm_pretraining(self):
        for pre_epoch in range(self.args.mlm_epochs):
            losses = AverageMeter('Loss', ':.4')
            progress = ProgressMeter(len(self.pre_train_loader), [losses], prefix="Epoch: [{}]".format(pre_epoch))
            for i, batch_dict in enumerate(self.pre_train_loader):
                self.backbone.train()
                if torch.cuda.is_available():
                    batch_dict = move_to_cuda(batch_dict)
                batch_size = len(batch_dict['labels'])

                outputs = self.backbone(**batch_dict)
                loss = outputs['loss']
                losses.update(loss.item(), batch_size)

                self.backbone_optimizer.zero_grad()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), self.args.grad_clip)

                self.backbone_optimizer.step()
                self.backbone_scheduler.step()
                # wandb.log({"pretraining-losses": losses.avg,})
                # wandb.log({"lr": self.backbone_scheduler.get_last_lr()[0]})

                if i % self.args.print_freq == 0:
                    progress.display(i)

            self.pretraining_eval_epoch(pre_epoch)

    def train_one_step(self, epoch, batch_dict):

        # switch to train mode
        self.model.train()

        if torch.cuda.is_available():
            _batch_dict = move_to_cuda(batch_dict)
        batch_size = len(_batch_dict['batch_data'])
        batch_dict = {}
        for k, v in _batch_dict.items(): 
            if k not in ['template', 'template_position_ids', 'template_attention_mask']:
                batch_dict[k] = v
        # compute output
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(**batch_dict)
        else:
            outputs = self.model(**batch_dict)
        outputs = get_model_obj(self.model).compute_logits(output_dict=outputs, batch_dict=batch_dict)
        outputs = ModelOutput(**outputs)

        logits, labels = outputs.logits, outputs.labels
        assert logits.size(0) == batch_size
        loss = self.criterion(logits, labels)
        loss += self.criterion(logits[:, :batch_size].t(), labels)

        acc1, acc3 = accuracy(logits, labels, topk=(1, 3))

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
        
        self.step += 1
        self.optimizer.step()
        self.scheduler.step()
        return acc1, acc3, outputs, loss.item()

    def _setup_training(self):
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model).cuda()
        elif torch.cuda.is_available():
            self.model.cuda()
        else:
            logger.info('No gpu will be used')


    def _create_lr_scheduler(self, num_training_steps):
        if self.args.lr_scheduler == 'linear':
            return get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                   num_warmup_steps=self.args.warmup,
                                                   num_training_steps=num_training_steps)
        elif self.args.lr_scheduler == 'cosine':
            return get_cosine_schedule_with_warmup(optimizer=self.optimizer,
                                                   num_warmup_steps=self.args.warmup,
                                                   num_training_steps=num_training_steps)
        else:
            assert False, 'Unknown lr scheduler: {}'.format(self.args.scheduler)
