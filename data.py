from torch.utils.data import Dataset
import pandas as pd
import json
from vocab import Vocab

class CommitDataset(Dataset):
    def __init__(self, src_vocab: Vocab, trg_vocab: Vocab, file_path):
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.df = pd.read_pickle(file_path)
    
    def __getitem__(self, index):
        src = json.loads(self.df["commit_message"][index])
        trg = json.loads(self.df["diff"][index])
        src_ids = self.src_vocab.tokens2ids(src)
        src_ids_ext = self.src_vocab.tokens2ids_ext(src)
        trg_ids = self.trg_vocab.tokens2ids(trg)
        return src_ids, src_ids_ext, trg_ids
    
    def __len__(self):
        return len(self.df)