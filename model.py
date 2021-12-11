from torch import nn
from torch.optim import Adagrad
from config import args
import torch.nn.functional as F
import torch
import pytorch_lightning as pl
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
"""
B : batch size
E : embedding size
H : encoder hidden state dimension
L : sequence length
T : target sequence length
"""

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
    def __init__(self, hidden_dim=args.hidden_dim):
        super().__init__()
        self.v = nn.Linear(hidden_dim * 2, 1, bias=False)                       # v
        self.enc_proj = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)   # W_h
        self.dec_proj = nn.Linear(hidden_dim, hidden_dim * 2, bias=True)        # W_s, b_attn
  

    def forward(self, dec_input, enc_hidden, enc_pad_mask):
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

        scores = self.v(torch.tanh(enc_feature + dec_feature)).squeeze(-1)  # [B X L]
        scores = scores.float().masked_fill_(
            enc_pad_mask,
            float('-inf')
        ).type_as(scores)  # FP16 support: cast to float and back
        
        attn_dist = F.softmax(scores, dim=-1) # [B X L]

        return attn_dist


class AttentionDecoderLayer(nn.Module):
  def __init__(self, input_dim, hidden_dim, vocab_size):
    super().__init__()
    self.lstm = nn.LSTMCell(input_size=input_dim, hidden_size=hidden_dim)
    self.attention = Attention(hidden_dim)
    self.l1 = nn.Linear(hidden_dim*3, hidden_dim, bias=True)    # V
    self.l2 = nn.Linear(hidden_dim, vocab_size, bias=True)  # V'
  
  def forward(self, dec_input, dec_hidden, dec_cell, enc_hidden, enc_pad_mask):
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
    attn_dist = self.attention(h, enc_hidden, enc_pad_mask)  # [B X 1 X L]
    context_vec = torch.bmm(attn_dist.unsqueeze(1), enc_hidden).squeeze(1)  # [B X 2H] <- [B X 1 X 2H] = [B X 1 X L] @ [B X L X 2H]
    output = self.l1(torch.cat([h, context_vec], dim = -1)) # [B X H]
    vocab_dist = F.softmax(self.l2(output), dim=-1)              # [B X V]
    return vocab_dist, attn_dist, context_vec, h, c


class PointerGenerator(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        embed_dim = args.embed_dim
        self.embedding = nn.Embedding(len(vocab), embed_dim, padding_idx=vocab.pad())


        hidden_dim = args.hidden_dim
        self.encoder = Encoder(input_dim=embed_dim, hidden_dim=hidden_dim)
        self.decoder = AttentionDecoderLayer(input_dim=embed_dim, hidden_dim=hidden_dim, vocab_size=len(vocab))

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
        enc_emb = self.src_embedding(enc_input)             # [B X L X E]
        enc_hidden, (h,c) = self.encoder(enc_emb, enc_len)  # [B X L X 2H], [B X L X H], [B X L X H]
        teacher_forcing = False

        if not dec_input is None:
            teacher_forcing = True
            dec_emb = self.embedding(dec_input)             # [B X T X E]
        else:
            dec_prev_emb = [self.embedding(self.vocab().start()) for _ in range(batch_size)]  # [B X E]


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
                enc_pad_mask=enc_pad_mask
            )
            
            p_gen = torch.sigmoid(self.w_h(context_vec) + self.w_s(h) + self.w_x(input_t))
            weighted_vocab_dist = p_gen * vocab_dist
            weighted_attn_dist = (1.0 - p_gen) * attn_dist
            B = vocab_dist.size(0)
            extended_vocab_dist = torch.cat([weighted_vocab_dist, torch.zeros((B, max_oov_len), device=vocab_dist.device)], dim=-1)

            final_dist = extended_vocab_dist.scatter_add(dim=-1, index=enc_input_ext, src=weighted_attn_dist) # index [B X L] source [B X VT]
            final_dists.append(final_dist)
            if (not teacher_forcing):
                highest_prob = torch.argmax(final_dist, dim=1)                              # [B]
                highest_prob[highest_prob >= len(self.vocab)] = self.vocab.unk()
                dec_prev_emb = self.embedding(B)        #[B X E]
        return torch.stack(final_dists, dim=-1)

class SummarizationModel(pl.LightningModule):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.model = PointerGenerator(vocab)
        self.num_step = 0
    
    def training_step(self, batch, batch_idx):
        output = self.model.forward(
            enc_input=batch.enc_input,
            enc_input_ext=batch.enc_input_ext,
            enc_pad_mask=batch.enc_pad_mask,
            enc_len=batch.enc_len,
            dec_input=batch.dec_input,
            max_oov_len=batch.max_oov_len)
        dec_target = batch.dec_target
        loss = F.nll_loss(torch.log(output), dec_target, ignore_index=args.pad_id, reduction='mean')
        self.logger.log_metrics({"train_loss": loss}, self.num_step)
        self.num_step += 1
        return loss
    
    def validation_step(self, batch, batch_idx):
        output = self.model.forward(
            enc_input=batch.enc_input,
            enc_input_ext=batch.enc_input_ext,
            enc_pad_mask=batch.enc_pad_mask,
            enc_len=batch.enc_len,
            dec_input=batch.dec_input,
            max_oov_len=batch.max_oov_len)
        
        dec_target = batch.dec_target
        loss = F.nll_loss(
            torch.log(output), dec_target, ignore_index=args.pad_id, reduction='mean')
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
        # TODO: FIXHERE
        result = {}
        result['target'] = output
        result['source'] = [' '.join(w) for w in batch.src_text]
        result['real_target'] = [' '.join(w) for w in batch.tgt_text]
        return result
    
    def configure_optimizers(self):
        return Adagrad(self.parameters(), lr=args.learning_rate, initial_accumulator_value=args.accum_init)

