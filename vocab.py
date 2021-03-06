'''
Reference https://github.com/jiminsun/pointer-generator/
'''
import numpy as np

pad_token = '<pad>'
unk_token = '<unk>'
start_decode = '<start>'
stop_decode = '<stop>'

class Vocab(object):
  def __init__(self):
    self._word_to_id = {}
    self._id_to_word = []
    self._count = 0

  @classmethod
  def from_counter(cls, counter, vocab_size, min_freq=1, specials=[pad_token, unk_token, start_decode, stop_decode]):
    vocab = cls()
    word_and_freq = sorted(counter.items(), key=lambda tup: tup[0])
    word_and_freq.sort(key=lambda tup: tup[1], reverse=True)

    for w in specials:
      vocab._word_to_id[w] = vocab._count
      vocab._id_to_word.append(w)
      vocab._count += 1

    for word, freq in word_and_freq:
      if freq < min_freq or vocab._count == vocab_size:
        break
      vocab._word_to_id[word] = vocab._count
      vocab._id_to_word.append(word)
      vocab._count += 1

    vocab.unk = vocab._word_to_id.get(unk_token)    
    vocab.pad = vocab._word_to_id.get(pad_token)
    vocab.start = vocab._word_to_id.get(start_decode)
    vocab.stop = vocab._word_to_id.get(stop_decode)
    return vocab
  
  def __len__(self):
    return self._count

  def word2id(self, word):
    unk_id = self._word_to_id.get(word, self.unk)
    if word in self._word_to_id:
      return self._word_to_id[word]
    else:
      return unk_id
  
  def id2word(self, word_id):
    if word_id >= self.__len__():
      raise ValueError(f"Id not found in vocab: {word_id}")
    return self._id_to_word[word_id]
  
  def tokens2ids(self, tokens):
    return np.array([self.word2id(t) for t in tokens])
  
  def tokens2ids_ext(self, tokens):
    ids = []
    oovs = []
    unk_id = self.unk
    for t in tokens:
      t_id = self.word2id(t)
      if t_id == unk_id:
        if t not in oovs:
          oovs.append(t)
        ids.append(len(self) + oovs.index(t))
      else:
        ids.append(t_id)
    return np.array(ids), oovs
  

  def tokens2ids_oovs(self, tokens, oovs):
    ids = []
    unk_id = self.unk
    for t in tokens:
      t_id = self.word2id(t)
      if t_id == unk_id:
        if t in oovs:
          ids.append(len(self)+oovs.index(t))
        else:
          ids.append(unk_id)
      else:
        ids.append(t_id)
    return np.array(ids)
  
  def ids2words_oovs(self, ids, oovs):
    tokens = []
    for id_ in ids:
      if (id_ >= len(self)):
        tokens.append(oovs[id_-len(self)])
      else:
        tokens.append(self.id2word(id_))
    
    return tokens
  