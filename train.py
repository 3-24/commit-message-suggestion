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

def train(root):
    pl.seed_everything(args.seed)

    src_counter = Counter()
    trg_counter = Counter()
    train_path = Path(root) / 'train.pkl'
    validation_path = Path(root) / 'validation.pkl'
    test_path = Path(root) / 'test.pkl'
    train_df = pd.read_pickle(train_path)

    for msg in train_df["diff"]:
        m = json.loads(msg)
        src_counter.update(m)

    for msg in train_df["commit_messsage"]:
        m = json.loads(msg)
        trg_counter.update(m)

    src_vocab = Vocab.from_counter(
        counter=src_counter, 
        vocab_size=args.vocab_size
    )

    trg_vocab = Vocab.from_counter(
        counter=trg_counter, 
        vocab_size=args.vocab_size
    )

    model = SummarizationModel(src_vocab, trg_vocab)

    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        max_epochs=args.epochs,
        gradient_clip_val=args.max_grad_norm
    )

    train_loader = DataLoader(
        CommitDataset(src_vocab, trg_vocab, train_path),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=commit_collate_fn
    )
    val_loader = DataLoader(
        CommitDataset(src_vocab, trg_vocab, validation_path),
        batch_size=args.batch_size,
        collate_fn=commit_collate_fn,
        shuffle=False
    )

    trainer.fit(model, train_loader, val_loader)

train('.')