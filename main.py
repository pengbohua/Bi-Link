import torch
import json
import torch.backends.cudnn as cudnn

import argparse
from trainer import Trainer, TrainingArguments
from cl_preprocess_data import EntityLinkingSet, load_documents
# from nsp_trainer import Trainer, TrainingArguments
# from preprocess_data import EntityLinkingSet, load_documents
from utils import logger
# import wandb

def get_args():
    parser = argparse.ArgumentParser("zero shot entity linker")


    parser.add_argument("--pretrained-model-path", default='/yinxr/hx/liangshihao/pretrained_models/bert-base-uncased/', type=str,
                        help="Path to pretrained transformers.")
    parser.add_argument("--eval-model-path", default='checkpoint', type=str,
                        help="Path to pretrained transformers.")
    parser.add_argument("--document-files", nargs="+", default=None,
                        help="Path to train documents json file.")
    parser.add_argument("--train-mentions-file", default=None, type=str,
                        help="Path to mentions json file.")
    parser.add_argument("--eval-mentions-file", default=None, type=str,
                        help="Path to mentions json file.")
    parser.add_argument("--train-tfidf-candidates-file", default='tfidf_candidates/train_tfidfs.json', type=str,
                        help="Path to TFIDF candidates file.")
    parser.add_argument("--eval-tfidf-candidates-file", default='tfidf_candidates/test_tfidfs.json', type=str,
                        help="Path to TFIDF candidates file.")
    parser.add_argument(
        "--split-by-domain", default=False, type=bool,
        help="Split output data file by domain.")
    parser.add_argument("--learning-rate", default=2e-5, type=float,
                        help="learning rate for optimization")
    parser.add_argument("--weight-decay", default=1e-4, type=float,
                        help="weight decay for optimization")
    parser.add_argument("--epochs", default=3, type=int,
                        help="weight decay for optimization")
    parser.add_argument("--train-batch-size", default=16, type=int,
                        help="train batch size")
    parser.add_argument("--eval-batch-size", default=128, type=int,
                        help="train batch size")
    parser.add_argument("--max-seq-length", default=64, type=int, help="Maximum sequence length.")

    parser.add_argument("--num-candidates", default=64, type=int, help="Number of tfidf candidates (0-63).")

    parser.add_argument("--random-seed", default=12345, type=int, help="Random seed for data generation.")

    parser.add_argument("--use-tf-idf-negatives", action="store_true", help="Use tf-idf as hard negatives in contrastive learning.")

    parser.add_argument("--use-mention-negatives", action="store_true", help="Use in-batch mention negatives as hard negatives in contrastive learning.")

    args = parser.parse_args()
    return args


def main():
    if torch.cuda.device_count() > 0:
        cudnn.benchmark = True

    # with wandb.init(settings=wandb.Settings(start_method="fork"), project="BMKG", entity="marvinpeng", config=vars(args)):
    #     wandb.config.update(args, allow_val_change=True)
    #     trainer = Trainer(args, ngpus_per_node=ngpus_per_node)
    # #     logger.info('Args={}'.format(json.dumps(args.__dict__, ensure_ascii=False, indent=4)))
    #     trainer.train_loop()

    args = get_args()
    args.use_rdrop = False
    train_args = TrainingArguments
    train_args.learning_rate = args.learning_rate
    train_args.train_batch_size = args.train_batch_size
    train_args.eval_batch_size = args.eval_batch_size
    train_args.num_cand = args.num_candidates
    train_args.epochs = args.epochs
    train_args.eval_every_n_intervals=2
    train_args.log_every_n_intervals=30

    all_documents = {}      # doc_id/ entity_id to entity
    document_path = args.document_files[0].split(",")
    for input_file_path in document_path:
        all_documents.update(load_documents(input_file_path))


    train_dataset = EntityLinkingSet(
                                    pretrained_model_path=args.pretrained_model_path,
                                    # document_files=args.document_files,
                                    document_files=all_documents,
                                     mentions_path=args.train_mentions_file,
                                     tfidf_candidates_file=args.train_tfidf_candidates_file,
                                     num_candidates=args.num_candidates,
                                     max_seq_length=args.max_seq_length,
                                     is_training=True)

    eval_dataset = EntityLinkingSet(
                                    pretrained_model_path=args.pretrained_model_path,
                                    # document_files=args.document_files,
                                    document_files=all_documents,
                                    mentions_path=args.eval_mentions_file,
                                   tfidf_candidates_file=args.eval_tfidf_candidates_file,
                                    num_candidates=args.num_candidates,
                                    max_seq_length=args.max_seq_length,
                                   is_training=False)

    print(args.use_rdrop)


    trainer = Trainer(
        pretrained_model_path=args.pretrained_model_path,
        eval_model_path=args.eval_model_path,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        train_args=train_args,
        use_tf_idf_negatives=args.use_tf_idf_negatives,
        use_in_batch_mention_negatives=args.use_mention_negatives,
        use_rdrop=args.use_rdrop,
        margin=args.margin
    )

    logger.info('Args={}'.format(json.dumps(args.__dict__, ensure_ascii=False, indent=4)))
    trainer.run()

if __name__ == '__main__':
    main()
