# Commit Message Suggestion

## Dataset

For all repositories in CodeSearchNet, we collected their diff and commit message. We provided [the notebook](dataset_generation/Dataset%20Generation.ipynb) that shows the overall process of dataset collection and tokenization. We also released the dataset at https://github.com/3-24/commit-message-suggestion/releases/tag/v1.0.0 which is composed of `train.pkl`, `validation.pkl`, `test.pkl`.  `pandas.read_pickle` allows you to read them as a pandas DataFrame object.

## Model

Model for commit message suggestion from code diffs. It is using pointer generator network architecture refering [Get to the Point: Summarization with Pointer-Generator Networks](https://research.google/pubs/pub46111/) and [its PyTorch implementations](https://github.com/jiminsun/pointer-generator).

## Install Requirements

```shell
pip install -r requirements.txt
```

## How to Use

`train.py` provides train function with four arguments. It trains the commit message suggestion with ten epochs.
```python
def train(root, use_pointer_gen=False, use_coverage=False, model_ckpt=None):
```
root is the path where the dataset is saved and train_pkl and validation_pkl should be prepared at that path. Set model_ckpt as the path to the pretrained model if you are training on pretrained models. If you are training the pointer generator network, use_pointer_gen should be True, and if you are training the PGN with coverage mechanism, both use_poitner_gen and use_coverage should be set as True. If you are training classical Sequence to Sequence model with attention mechanism, set use_pointer_gen=False and use_coverage=False.

`test.py` is similar. It provides test function with four arguments and it evaluates the model by using various metrics such as ROGUE and BLEU. in this case, model_ckpt is the path of the model to evaluate and this is always required. Other arguments is same as train.
