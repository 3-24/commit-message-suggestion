from torch.utils.data import Dataset
import pandas as pd
import json
from vocab import Vocab
import torch
from config import args
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad
import numpy as np

class CommitDataset(Dataset):
    def __init__(self, vocab: Vocab, file_path):
        self.vocab = vocab
        df = pd.read_pickle(file_path)
        self.df = df[df.apply(lambda x: len(json.loads(x["diff"]))<args.src_max_len, axis=1)]
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        src = json.loads(row["diff"])
        trg = json.loads(row["commit_messsage"])
        src_ids = self.vocab.tokens2ids(src)
        src_ids_ext, oovs = self.vocab.tokens2ids_ext(src)
        trg_ids = self.vocab.tokens2ids(trg)
        trg_ids_ext = self.vocab.tokens2ids_oovs(trg, oovs)
        return src, trg, src_ids, src_ids_ext, oovs, trg_ids, trg_ids_ext
    
    def __len__(self):
        return len(self.df)


class Batch:
    def __init__(self, batchdata):
        srcs, trgs, src_ids, src_ids_ext, oovs, trg_ids, trg_ids_ext = zip(*batchdata)
        self.enc_input = pad_sequence(tuple(map(torch.LongTensor, src_ids)), batch_first=True, padding_value=0)
        self.enc_input_ext = pad_sequence(tuple(map(torch.LongTensor, src_ids_ext)), batch_first=True, padding_value=0)
        self.enc_pad_mask = (self.enc_input == 0)
        self.enc_len = torch.LongTensor([len(tokens) for tokens in src_ids])
        self.dec_input = collate_tokens(values=map(lambda x: torch.LongTensor(np.insert(x, 0, 2)), trg_ids),
                                        pad_idx=0,
                                        pad_to_length=args.trg_max_len)
        self.dec_target = collate_tokens(values=map(lambda x: torch.LongTensor(np.append(x, 3)), trg_ids_ext),
                                        pad_idx=0,
                                        pad_to_length=args.trg_max_len)
        self.dec_len = torch.LongTensor([len(tokens)+1 for tokens in trg_ids])    # +1 for '<stop>' token
        self.max_oov_len = max(map(len, oovs))
        self.src_text = srcs
        self.trg_text = trgs
        self.oovs = oovs

    
    def to(self, device):
        self.enc_input = self.enc_input.to(device)
        self.enc_input_ext = self.enc_input_ext.to(device)
        self.enc_pad_mask = self.enc_pad_mask.to(device)
        self.enc_len = self.enc_len.to(device)
        self.dec_input = self.dec_input.to(device)
        self.dec_target = self.dec_target.to(device)
        self.dec_len = self.dec_len.to(device)

def commit_collate_fn(batchdata):
    return Batch(batchdata)

'''
Reference: https://github.com/jiminsun/pointer-generator/blob/8ef1633f2fa66d7a51378caa7cfd72e48e7fc513/data/utils.py
'''
def collate_tokens(values, pad_idx, left_pad=False,
                   pad_to_length=None):
    # Simplified version of `collate_tokens` from fairseq.data.data_utils
    """Convert a list of 1d tensors into a padded 2d tensor."""
    values = list(map(torch.LongTensor, values))
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res