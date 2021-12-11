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
        trg[0:0] = [self.trg_vocab.start()]
        trg.append(self.trg_vocab.stop())
        item.src_ids = self.src_vocab.tokens2ids(src)
        item.src_ids_ext, item.oovs = self.src_vocab.tokens2ids_ext(src)
        item.trg_ids = self.trg_vocab.tokens2ids(trg)
        
        return item
    
    def __len__(self):
        return len(self.df)


def commit_collate_fn(batchdata):
    src_ids = [b.src_ids for b in batchdata]
    src_ext_ids = [b.src_ids_ext for b in batchdata]
    oovs = [b.oovs for b in batchdata]
    trg_ids = [b.trg_ids for b in batchdata]
    print(batchdata)
    print(len(batchdata))
    batch = EasyDict()
    batch.max_oov_len = max([len(oov) for oov in oovs])
    assert(False)
    '''
    batch = EasyDict()
    batch.enc_input
    batch.enc_input_ext
    batch.enc_pad_mask
    batch.enc_len
    batch.dec_input
  '''