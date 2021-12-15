'''
Highly inspired by https://github.com/jiminsun/pointer-generator
'''

from torch import nn
from torch.optim import Adagrad, Adam
from config import args
import torch.nn.functional as F
import torch
import pytorch_lightning as pl
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from itertools import chain
from utils import Metric
"""
B : batch size
E : embedding size
H : encoder hidden state dimension
L : sequence length
T : target sequence length
"""

debug = False

class Encoder(nn.Module):

    def __init__(self, input_dim=args.embed_dim, hidden_dim=args.hidden_dim):
        """
        Args:
            input_dim: source embedding dimension
        """
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.reduce_h = nn.Linear(hidden_dim * 2, hidden_dim, bias=True)
        self.reduce_c = nn.Linear(hidden_dim * 2, hidden_dim, bias=True)
    
    def forward(self, src, src_lens):
        """
        Args:
            src: source token embeddings    [B x L x E]
            src_lens: source text length    [B]
        Returns:
            enc_hidden: sequence of encoder hidden states                  [B x L x 2H]
            (final_h, final_c): Tuple for decoder state initialization     [B x L x H]
        """

        x = pack_padded_sequence(src, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        output, (h, c) = self.lstm(x) # [B x L x 2H], [2 x B x H], [2 x B x H]
        enc_hidden, _ = pad_packed_sequence(output, batch_first=True)

        # Concatenate bidirectional lstm states
        h = torch.cat((h[0], h[1]), dim=-1)  # [B x 2H]
        c = torch.cat((c[0], c[1]), dim=-1)  # [B x 2H]

        # Project to decoder hidden state size
        final_hidden = torch.relu(self.reduce_h(h))  # [B x H]
        final_cell = torch.relu(self.reduce_c(c))  # [B x H]

        return enc_hidden, (final_hidden, final_cell)


class Attention(nn.Module):
    def __init__(self, hidden_dim=args.hidden_dim, use_coverage=False):
        super().__init__()
        self.v = nn.Linear(hidden_dim * 2, 1, bias=False)                       # v
        self.enc_proj = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)   # W_h
        self.dec_proj = nn.Linear(hidden_dim, hidden_dim * 2, bias=True)        # W_s, b_attn

        self.use_coverage = use_coverage
        if (use_coverage):
            self.w_c = nn.Linear(1, hidden_dim * 2, bias=False) 

    def forward(self, dec_input, enc_hidden, enc_pad_mask, coverage=None):
        """
        Args:
            dec_input: decoder hidden state             [B x H]
            enc_hidden: encoder hidden states           [B x L x 2H]
            enc_pad_mask: encoder padding masks         [B x L]
        Returns:
            attn_dist: attention dist'n over src tokens [B x L]
        """
        enc_feature = self.enc_proj(enc_hidden)               # [B X L X 2H]
        dec_feature = self.dec_proj(dec_input).unsqueeze(1)   # [B X 1 X 2H]
        scores = enc_feature +dec_feature

        if self.use_coverage:
            coverage = coverage.unsqueeze(-1)               # [B X L X 1]
            scores = scores + self.w_c(coverage)

        scores = self.v(torch.tanh(scores)).squeeze(-1)  # [B X L]
        scores = scores.float().masked_fill_(
            enc_pad_mask,
            float('-inf')
        ).type_as(scores)
        
        attn_dist = F.softmax(scores, dim=-1) # [B X L]

        return attn_dist


class AttentionDecoderLayer(nn.Module):
  def __init__(self, input_dim, hidden_dim, vocab_size, use_coverage=False):
    super().__init__()
    self.use_coverage = use_coverage
    self.lstm = nn.LSTMCell(input_size=input_dim, hidden_size=hidden_dim)
    self.attention = Attention(hidden_dim, use_coverage=use_coverage)
    self.l1 = nn.Linear(hidden_dim*3, hidden_dim, bias=True)    # V
    self.l2 = nn.Linear(hidden_dim, vocab_size, bias=True)  # V'
  
  def forward(self, dec_input, dec_hidden, dec_cell, enc_hidden, enc_pad_mask, coverage=None):
    """
    Args:
        dec_input: decoder input embedding at timestep t    [B x E]
        prev_h: decoder hidden state from prev timestep     [B x H]
        prev_c: decoder cell state from prev timestep       [B x H]
        enc_hidden: encoder hidden states                   [B x L x 2H]
        enc_pad_mask: encoder masks for attn computation    [B x L]
    Returns:
        vocab_dist: predicted vocab dist'n at timestep t    [B x V]
        attn_dist: attention dist'n at timestep t           [B x L]
        context_vec: context vector at timestep t           [B x 2H]
        hidden: hidden state at timestep t                  [B x H]
        cell: cell state at timestep t                      [B x H]
    """
    h, c = self.lstm(dec_input, (dec_hidden, dec_cell))  # [B X H], [B X H]
    attn_dist = self.attention(h, enc_hidden, enc_pad_mask, coverage=coverage)  # [B X 1 X L]
    context_vec = torch.bmm(attn_dist.unsqueeze(1), enc_hidden).squeeze(1)  # [B X 2H] <- [B X 1 X 2H] = [B X 1 X L] @ [B X L X 2H]
    output = self.l1(torch.cat([h, context_vec], dim = -1)) # [B X H]
    vocab_dist = F.softmax(self.l2(output), dim=-1)              # [B X V]
    return vocab_dist, attn_dist, context_vec, h, c


class Seq2SeqAttn(nn.Module):
    def __init__(self, vocab, use_pointer_gen=False, use_coverage=False):
        super().__init__()
        self.use_pointer_gen = use_pointer_gen
        self.use_coverage = use_coverage
        self.vocab = vocab
        embed_dim = args.embed_dim
        self.embedding = nn.Embedding(len(vocab), embed_dim, padding_idx=vocab.pad())

        hidden_dim = args.hidden_dim
        self.encoder = Encoder(input_dim=embed_dim, hidden_dim=hidden_dim)
        self.decoder = AttentionDecoderLayer(input_dim=embed_dim, hidden_dim=hidden_dim, vocab_size=len(vocab), use_coverage=use_coverage)
        if use_pointer_gen:
            self.w_h = nn.Linear(hidden_dim * 2, 1, bias=False)
            self.w_s = nn.Linear(hidden_dim, 1, bias=False)
            self.w_x = nn.Linear(embed_dim, 1, bias=True)


    def forward(self, enc_input, enc_input_ext, enc_pad_mask, enc_len, max_oov_len, dec_input=None):
        """
        Predict summary using reference summary as decoder inputs. If dec_input is not provided, then teacher forcing is disabled.
        Args:
            enc_input: source text id sequence                      [B x L]
            enc_input_ext: source text id seq w/ extended vocab     [B x L]
            enc_pad_mask: source text padding mask. [PAD] -> True   [B x L]
            enc_len: source text length                             [B]
            dec_input: target text id sequence                      [B x T]
            max_oov_len: max number of oovs in src                  [1]
        Returns:
            final_dists: predicted dist'n using extended vocab      [B x V_x x T]
            attn_dists: attn dist'n from each t                     [B x L x T]
            coverages: coverage vectors from each t                 [B x L x T]
        """
        batch_size = enc_input.size(0)
        enc_emb = self.embedding(enc_input)             # [B X L X E]
        enc_hidden, (h,c) = self.encoder(enc_emb, enc_len)  # [B X L X 2H], [B X L X H], [B X L X H]
        teacher_forcing = False

        if self.use_coverage:
            cov = torch.zeros_like(enc_input).float()
            coverages = []
            attns = []

        if not dec_input is None:
            teacher_forcing = True
            dec_emb = self.embedding(dec_input)             # [B X T X E]
        else:
            dec_prev_emb = self.embedding(torch.full((batch_size,), self.vocab.start(), device=enc_emb.device))

        final_dists = []

        for t in range(args.trg_max_len):
            if teacher_forcing:
                input_t = dec_emb[:, t, :]
            else:
                input_t = dec_prev_emb
            vocab_dist, attn_dist, context_vec, h, c = self.decoder(
                dec_input=input_t, # [B x E]
                dec_hidden=h,
                dec_cell=c,
                enc_hidden=enc_hidden,
                enc_pad_mask=enc_pad_mask,
                coverage=(None if not self.use_coverage else cov)
            )
            if self.use_coverage:
                cov = cov + attn_dist
                coverages.append(cov)
                attns.append(attn_dist)

            if self.use_pointer_gen:
                p_gen = torch.sigmoid(self.w_h(context_vec) + self.w_s(h) + self.w_x(input_t))
                weighted_vocab_dist = p_gen * vocab_dist
                weighted_attn_dist = (1.0 - p_gen) * attn_dist
                extended_vocab_dist = torch.cat([weighted_vocab_dist, torch.zeros((batch_size, max_oov_len), device=vocab_dist.device)], dim=-1)
                final_dist = extended_vocab_dist.scatter_add(dim=-1, index=enc_input_ext, src=weighted_attn_dist) # index [B X L] source [B X VT]
            else:
                final_dist = vocab_dist
            final_dists.append(final_dist)

            if (not teacher_forcing):
                highest_prob = torch.argmax(final_dist, dim=1)                              # [B]
                highest_prob[highest_prob >= len(self.vocab)] = self.vocab.unk()
                dec_prev_emb = self.embedding(highest_prob)        #[B X E]
        
        if self.use_coverage:
            return {"final_dist": torch.stack(final_dists, dim=-1), "attn_dist": torch.stack(attns, dim=-1), "coverage": torch.stack(coverages, dim=-1)}
        else:
            return {"final_dist": torch.stack(final_dists, dim=-1)}


class SummarizationModel(pl.LightningModule):
    def __init__(self, vocab, use_pointer_gen=False, use_coverage=False):
        super().__init__()
        self.use_pointer_gen = use_pointer_gen
        self.use_coverage = use_coverage
        self.vocab = vocab
        self.model = Seq2SeqAttn(vocab, use_pointer_gen=use_pointer_gen, use_coverage=use_coverage)
        self.num_step = 0
    
    def get_loss(self, batch, inference=False):
        output = self.model.forward(
            enc_input=batch.enc_input,
            enc_input_ext=batch.enc_input_ext,
            enc_pad_mask=batch.enc_pad_mask,
            enc_len=batch.enc_len,
            dec_input=(None if inference else batch.dec_input),
            max_oov_len=batch.max_oov_len)
        
        final_dist = output["final_dist"]   # [B X V X T]
        batch_size = final_dist.size(0)
        dec_target = batch.dec_target       # [B X T]
        # final_dist = final_dist.masked_fill_((batch.dec_target.unsqueeze(1) == 0), 1.0)
              
        if (not self.use_pointer_gen):
            dec_target[dec_target >= len(self.vocab)] = self.vocab.unk()
        
        loss = F.nll_loss(torch.log(final_dist), dec_target, ignore_index=args.pad_id, reduction='mean')
        
        if (self.use_coverage):
            cov_loss = torch.sum(torch.min(output["attn_dist"], output["coverage"]), dim=1) # [B X T]
            cov_loss = cov_loss.masked_fill_((batch.dec_target == 0), 0.0)
            cov_loss = torch.sum(cov_loss, dim=1) / batch.dec_len  # [B]
            loss = loss + torch.sum(cov_loss) / batch_size
        
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.logger.log_metrics({"train_loss": loss}, self.num_step)
        self.num_step += 1
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.get_loss(batch, inference=True)
        self.log('val_loss', loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.logger.log_metrics({'val_loss': loss}, self.num_step)
        return loss

    def test_step(self, batch, batch_idx):
        output = self.model.forward(
            enc_input=batch.enc_input,
            enc_input_ext=batch.enc_input_ext,
            enc_pad_mask=batch.enc_pad_mask,
            enc_len=batch.enc_len,
            max_oov_len=batch.max_oov_len
        )
        
        def _postprocess(highest_prob, oov):
            tokens = self.vocab.ids2words_oovs(highest_prob, oov)
                
            stop_idx = tokens.index('<stop>') if '<stop>' in tokens else len(tokens)
            return  tokens[:stop_idx]

        final_dists = output["final_dist"]
        highest_probs = torch.argmax(final_dists, dim=1)
        result = {}
        result['gen_target'] = [_postprocess(hp, oov) for hp, oov in zip(torch.unbind(highest_probs), batch.oovs)]
        result['source'] = batch.src_text       #[' '.join(src) for src in batch.src_text]
        result['real_target'] = batch.trg_text  #[' '.join(trg) for trg in batch.trg_text]
        if debug:
            if result['gen_target'][0][0] =="fix" or True:
                print(' '.join(result['source'][0]))
                print("generated commit message : ",' '.join(result['gen_target'][0]))
                print("real commit message      : ", ' '.join(result['real_target'][0]))
                print(batch.indices)
                input()
        return result
    
    def test_epoch_end(self, test_output):
        gen_commits = list(chain.from_iterable([batch["gen_target"] for batch in test_output]))
        real_commits = list(chain.from_iterable([batch["real_target"] for batch in test_output]))
        rouge_score = Metric.get_rouge_score(gen_commits,real_commits)
        duplicate_score = Metric.duplicate_vocab(gen_commits)
        BLEU_score = Metric.get_BLEU(gen_commits,real_commits)
        result = {"rouge-1":rouge_score[0],"rouge-2":rouge_score[1],"rouge-l":rouge_score[2],"duplicate_rate":duplicate_score,"BLEU":BLEU_score}
        self.log_dict(result)
        return result
    
    def configure_optimizers(self):
        return Adagrad(self.parameters(), lr=args.learning_rate, initial_accumulator_value=args.accum_init)
        #return Adam(self.parameters(), lr=args.learning_rate)