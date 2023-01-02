import torch
import json
import torch.backends.cudnn as cudnn

from config import args
# from trainer import Trainer
from trainer import Trainer
from logger_config import logger
import wandb
from utils import seed_everything

def main():
    ngpus_per_node = torch.cuda.device_count()
    cudnn.benchmark = True
    seed_everything(1234)

    logger.info("Use {} gpus for training".format(ngpus_per_node))
    with wandb.init(settings=wandb.Settings(start_method="fork"), project="bilink", entity="marvinpeng", config=vars(args)):
        wandb.config.update(args, allow_val_change=True)
        trainer = Trainer(args)
        logger.info('Args={}'.format(json.dumps(args.__dict__, ensure_ascii=False, indent=4)))
        trainer.train_loop()

 

if __name__ == '__main__':
    main()
