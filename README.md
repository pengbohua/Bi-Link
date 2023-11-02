# Bi-Link
Code of Bi-Link will be updated in this repository.
## Data Preparation
Extract knowledge graphs to the data folder similar as follows:
```python
wiki5m_ind
├── train.txt
├── valid.txt
├── test.txt
├── wikidata5m_entity.txt
├── wikidata5m_relation.txt
└── wikidata5m_text.txt 
```

## Data Preprocessing
```bash
bash scripts/preprocess.sh wiki5m_ind
```
## Train
```bash
bash scripts/train.sh 
```
## Evaluation
Check DATA_DIR and model_path. Run the following evaluation snippet.
```bash
bash scripts/eval.sh
bash scripts/eval_wiki5m_trans.sh
```
## Checkpoints
| Datasets                                                                                                                                                             | Checkpoints                                                          |
|-------------------------|----------------------------------------------------------------------|
| WN18RR                  | [Checkpoint](https://mega.nz/folder/8HMw2KJR#iGgjtjyd0CX92rKs656P5g) |
| Wikidata5M-transductive | [Checkpoint](https://mega.nz/folder/ob8mXYoL#1YXiUlX8RI7NZdrAnvypdA) |

