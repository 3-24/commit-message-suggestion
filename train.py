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
from pytorch_lightning.callbacks import ModelCheckpoint

def train(root):
    pl.seed_everything(args.seed)

    counter = Counter()
    train_path = Path(root) / 'train.pkl'
    validation_path = Path(root) / 'validation.pkl'
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

    model = SummarizationModel(vocab)

    checkpoint_callback = ModelCheckpoint(dirpath=f"{root}/checkpoints/")

    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        max_epochs=args.epochs,
        gradient_clip_val=args.max_grad_norm,
        callbacks=[checkpoint_callback],
        precision=16
    )

    train_loader = DataLoader(
        CommitDataset(vocab, train_path),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=commit_collate_fn
    )
    val_loader = DataLoader(
        CommitDataset(vocab, validation_path),
        batch_size=args.batch_size,
        collate_fn=commit_collate_fn,
        shuffle=False
    )

    trainer.fit(model, train_loader, val_loader)