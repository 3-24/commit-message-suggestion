import pytorch_lightning as pl
from config import args
from vocab import Vocab
from pathlib import Path
from collections import Counter
import pandas as pd
import json
import torch
from model import SummarizationModel
from data import CommitDataset, commit_collate_fn
from torch.utils.data import DataLoader

def test(root, model_ckpt, pointer_gen=False, coverage=False):
    pl.seed_everything(args.seed)
    counter = Counter()
    train_path = Path(root) / 'train.pkl'
    test_path = Path(root) / 'test.pkl'
    train_df = pd.read_pickle(train_path)

    for msg in train_df["diff"]:
        m = json.loads(msg)
        counter.update(m)

    for msg in train_df["commit_messsage"]:
        m = json.loads(msg)
        counter.update(m)

    vocab = Vocab.from_counter(
        counter=counter, 
        vocab_size=args.vocab_size
    )

    model = SummarizationModel.load_from_checkpoint(vocab=vocab, checkpoint_path=model_ckpt, poitner_gen=pointer_gen, coverage=coverage)
    model.freeze()
    model.eval()

    trainer = pl.Trainer(
        gpus=torch.cuda.device_count()
    )

    test_loader = DataLoader(
        CommitDataset(vocab, test_path),
        batch_size=1,
        collate_fn=commit_collate_fn,
        shuffle=False
    )

    trainer.test(model=model, ckpt_path=model_ckpt, dataloaders=test_loader)

test('.')