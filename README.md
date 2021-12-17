# Commit Message Suggestion

## Dataset

For all repositories in CodeSearchNet, we collected their diff and commit message. We provided [the notebook](dataset_generation/Dataset%20Generation.ipynb) that shows the overall process of dataset collection and tokenization. We also released the dataset at https://github.com/3-24/commit-message-suggestion/releases/tag/v1.0.0which is composed of `train.pkl`, `validation.pkl`, `test.pkl`.  `pandas.read_pickle` allows you to read them as a pandas DataFrame object.

## Model

Model for commit message suggestion from code diffs. It is using pointer generator network architecture refering [Get to the Point: Summarization with Pointer-Generator Networks](https://research.google/pubs/pub46111/) and [its PyTorch implementations](https://github.com/jiminsun/pointer-generator).

