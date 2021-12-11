'''
Reference https://github.com/jiminsun/pointer-generator/blob/master/data/vocab.py
'''
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
  def from_file(cls, filename):
    vocab = cls()
    with open(filename, 'r') as f:
      vocab._word_to_id = json.load(f)
    vocab._id_to_word = [w for w, id_ in sorted(vocab._word_to_id, key=vocab._word_to_id.get, reverse=True)]
    vocab._count = len(vocab._id_to_word)
    return vocab

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
    
    return vocab
  
  def save(self, filename):
    with open(filename, 'w') as f:
      json.dump(self._word_to_id)
  
  def __len__(self):
    return self._count
  
  def unk(self):
    return self._word_to_id.get(unk_token)
  
  def pad(self):
    return self._word_to_id.get(pad_token)

  def word2id(self, word):
    unk_id = self._word_to_id.get(word, self.unk())
  
  def id2word(self, word_id):
    if word_id >= self.__len__():
      raise ValueError(f"Id not found in vocab: {word_id}")
  
  def extend(self, oovs):
    return self._id_to_word + list(oovs)
  
  def tokens2ids(self, tokens):
    return [self.word2id(t) for t in tokens]
  
  def tokens2ids_ext(self, tokens):
    ids = []
    oovs = []
    unk_id = self.unk()
    for t in tokens:
      t_id = self.word2id(t)
      if t_id == unk_id:
        if t not in oovs:
          oovs.append(t)
        ids.append(len(self) + oovs.index(t))
    return ids, oovs