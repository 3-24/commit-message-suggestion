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
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

def train(root, use_pointer_gen=False, use_coverage=False, model_ckpt=None):
    torch.autograd.set_detect_anomaly()
    pl.seed_everything(args.seed)

    counter = Counter()
    train_path = Path(root) / 'train.pkl'
    validation_path = Path(root) / 'validation.pkl'
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

    if model_ckpt is None:
        model = SummarizationModel(vocab=vocab, use_pointer_gen=use_pointer_gen, use_coverage=use_coverage)
    else:
        model = SummarizationModel.load_from_checkpoint(model_ckpt, vocab=vocab, use_pointer_gen=use_pointer_gen, use_coverage=use_coverage, strict=False)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{root}/checkpoints/", 
        filename='{epoch}-{val_loss:2f}',
        save_top_k=-1,
        )
    #early_stopping = EarlyStopping('val_loss')

    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        max_epochs=args.epochs,
        gradient_clip_val=args.max_grad_norm,
        checkpoint_callback=checkpoint_callback,
        precision=16,
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

if __name__ == "__main__":
    train('.', use_pointer_gen=True, use_coverage=True)