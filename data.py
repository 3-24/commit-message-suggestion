from torch.utils.data import Dataset
import pandas as pd
import json
from vocab import Vocab
from easydict import EasyDict
import torch
from config import args
from torch.nn.utils.rnn import pad_sequence

class CommitDataset(Dataset):
    def __init__(self, vocab: Vocab, file_path):
        self.vocab = vocab
        df = pd.read_pickle(file_path)
        self.df = df[df.apply(lambda x: len(json.loads(x["diff"]))<args.src_max_len, axis=1)]
    
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


class Batch:
  def __init__(self, batchdata):
      src_ids, src_ids_ext, oovs, trg_ids = zip(*batchdata)
      self.enc_input = pad_sequence(tuple(map(torch.LongTensor, src_ids)), batch_first=True, padding_value=0)
      self.enc_input_ext = pad_sequence(tuple(map(torch.LongTensor, src_ids_ext)), batch_first=True, padding_value=0)
      self.enc_pad_mask = (self.enc_input == 0)
      self.enc_len = torch.LongTensor([len(tokens) for tokens in src_ids])
      self.dec_input = pad_sequence(tuple(map(lambda x: torch.LongTensor(np.insert(x, 0, 2)), trg_ids)), batch_first=True, padding_value=0)
      self.dec_target = pad_sequence(tuple(map(lambda x: torch.LongTensor(np.append(x, 3)), trg_ids)), batch_first=True, padding_value=0)
      self.max_oov_len = max(map(len, oovs))
    
  def to(self, device):
    self.enc_input = self.enc_input.to(device)
    self.enc_input_ext = self.enc_input_ext.to(device)
    self.enc_pad_mask = self.enc_pad_mask.to(device)
    self.enc_len = self.enc_len.to(device)
    self.dec_input = self.dec_input.to(device)
    self.dec_target = self.dec_target.to(device)

def CommitCollate(batchdata):
    return Batch(batchdata)