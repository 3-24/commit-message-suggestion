import pytorch_lightning as pl
from config import args
from vocab import Vocab
from pathlib import Path
from collections import Counter
import pandas as pd
import json

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

