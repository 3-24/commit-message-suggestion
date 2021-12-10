import pytorch_lightning as pl
from config import args
from vocab import Vocab
from pathlib import Path
from collections import Counter
import pandas as pd
import json
import torch
from model import SummarizationModel

root = '.'

pl.seed_everything(args.seed)

src_counter = Counter()
trg_counter = Counter()
data_path = Path(root) / 'validation.pkl'   # Replace later
train_df = pd.read_pickle(data_path)

for msg in train_df["diff"]:
  m = json.loads(msg)
  src_counter.update(m)

for msg in train_df["commit_messsage"]:
  m = json.loads(msg)
  trg_counter.update(m)

src_vocab = Vocab.from_counter(
    counter=src_counter, 
    vocab_size=args.vocab_size,
    min_freq=2
)

trg_vocab = Vocab.from_counter(
    counter=trg_counter, 
    vocab_size=args.vocab_size,
    min_freq=2
)

model = SummarizationModel(src_vocab, trg_vocab)

trainer = pl.Trainer(
    gpus=torch.cuda.device_count(),
    max_epochs=args.epochs,
    gradient_clip_val=args.max_grad_norm
    )

#trainer.fit(model, train_loader, val_loader)

