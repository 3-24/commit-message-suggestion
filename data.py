from torch.utils.data import Dataset
import pandas as pd
import json
from vocab import Vocab
from easydict import EasyDict
import torch

class CommitDataset(Dataset):
    def __init__(self, vocab: Vocab, file_path):
        self.vocab = vocab
        self.df = pd.read_pickle(file_path)
    
    def __getitem__(self, index):
        item = EasyDict()
        row = self.df.iloc[index]
        src = json.loads(row["diff"])
        trg = json.loads(row["commit_messsage"])
        item.src_ids = self.vocab.tokens2ids(src)
        item.src_ids_ext, item.oovs = self.vocab.tokens2ids_ext(src)
        item.trg_ids = self.vocab.tokens2ids(trg)
        return item
    
    def __len__(self):
        return len(self.df)


def commit_collate_fn(batchdata):
    size = len(batchdata)
    max_enc_len, max_dec_len, max_oov_len = 0,0,0
    enc_len_list = [len(batchdata[i]['src_ids']) for i in range(size)]
    trg_ids = [batchdata[i]['trg_ids'] for i in range(size)]
    for i in range(size):
        max_enc_len = max(len(batchdata[i]['src_ids']), max_enc_len)
        max_dec_len = max(len(batchdata[i]['trg_ids']), max_dec_len)
        max_oov_len = max(len(batchdata[i]['oovs']), max_oov_len)

    for i in range(len(batchdata)):
        batchdata[i]['src_ids'] += [0]*(max_enc_len-len(batchdata[i]['src_ids']))
        batchdata[i]['src_ids_ext'] += [0]*(max_enc_len-len(batchdata[i]['src_ids_ext']))

    batch = EasyDict()
    batch.enc_input = torch.LongTensor([batchdata[i]['src_ids'] for i in range(size)])
    batch.enc_input_ext = torch.LongTensor([batchdata[i]['src_ids_ext'] for i in range(size)])
    batch.enc_pad_mask = (batch.enc_input == 0)
    batch.enc_len = torch.LongTensor(enc_len_list)
    batch.dec_input = torch.LongTensor([[2] + trg_ids[i] + [0]*(max_dec_len-len(trg_ids[i])) for i in range(size)])
    batch.dec_target = torch.LongTensor([trg_ids[i] + [3] + [0]*(max_dec_len-len(trg_ids[i])) for i in range(size)])
    batch.max_oov_len = max_oov_len
    return batch
