from torch.utils.data import Dataset
import pandas as pd
import json
from vocab import Vocab
from easydict import EasyDict

class CommitDataset(Dataset):
    def __init__(self, src_vocab: Vocab, trg_vocab: Vocab, file_path):
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.df = pd.read_pickle(file_path)
    
    def __getitem__(self, index):
        item = EasyDict()
        row = self.df.iloc[index]
        src = json.loads(row["commit_messsage"])
        trg = json.loads(row["diff"])
        item.src_ids = self.src_vocab.tokens2ids(src)
        item.src_ids_ext, item.oovs = self.src_vocab.tokens2ids_ext(src)
        item.trg_ids = self.trg_vocab.tokens2ids(trg)
        
        return item
    
    def __len__(self):
        return len(self.df)