import torch.nn.functional as F
import torch.nn as nn
import torch

class Causal_LM(nn.Module):
    def __init__(self, d_model, num_head, n_position, vocab_size, pad_id, device):
        super().__init__()
        self.posit_table = nn.Embedding(n_position, d_model)
        self.embed_table = nn.Embedding(vocab_size, d_model)
        self.mh1 = nn.MultiheadAttention(d_model, num_head)
        self.mh2 = nn.MultiheadAttention(d_model, num_head)
        self.fc1 = nn.Linear(in_features=d_model, out_features=d_model*2)
        self.fc2 = nn.Linear(in_features=d_model*2, out_features=d_model)
        self.loss_fct = nn.CrossEntropyLoss()
        self.device = device
        self.pad_id = pad_id

    def forward(self, seq):
        posit_ids = torch.arange(0, seq["input_ids"].size()[-1], device=self.device)
        posit_ids = posit_ids.repeat(seq["input_ids"].size()[0], 1)
        posit_embeds = self.posit_table(posit_ids)
        input_embeds = self.embed_table(seq["input_ids"])
        attn_mask = self.generate_square_subsequent_mask(seq["input_ids"].size()[-1])
        kpad_mask = torch.where(seq["attention_mask"]==0, True, False)
        labels = torch.where(seq["input_ids"]==self.pad_id, torch.tensor(-100).to(self.device), seq["input_ids"])

        qkv = (posit_embeds + input_embeds).permute([1, 0, 2])    # N len C → len N C
        in1 = F.layer_norm(qkv, qkv.size()[1:])
        ou1, _ = self.mh1(in1, in1, in1, key_padding_mask=kpad_mask, need_weights=False, attn_mask=attn_mask)
        ffo = self.fc2(F.leaky_relu(self.fc1(ou1 + qkv))) + (ou1 + qkv)
        in2 = F.layer_norm(ffo, ffo.size()[1:])
        ou2, _ = self.mh2(in2, in2, in2, key_padding_mask=kpad_mask, need_weights=False, attn_mask=attn_mask)
        output = ou2.permute([1, 0, 2])    # len N C → N len C
        lm_logits = torch.matmul(output, self.embed_table.weight.permute([1, 0]))    # N len C * C V → N len V

        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss

    def generate_square_subsequent_mask(self, sz: int):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1).to(self.device)


class EncoderRNN(nn.Module):
    def __init__(self, embed_table, hidden_size):
        super().__init__()
        self.embed_table = embed_table
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, seq, hidden=None):
        input_embeds = self.embed_table(seq["input_ids"])
        lengths = torch.sum(seq["attention_mask"], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(input_embeds, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed, hidden)

        return hidden


class DecoderRNN(nn.Module):
    def __init__(self, embed_table, hidden_size, vocab_size):
        super().__init__()
        self.embed_table = embed_table
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, seq, last_hidden):
        input_embeds = self.embed_table(seq["input_ids"])
        lengths = torch.sum(seq["attention_mask"], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(input_embeds, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, _ = self.gru(packed, last_hidden)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        logits = self.out(output)  # N Len V

        return logits


class AutoEncoder(nn.Module):
    def __init__(self, d_model, vocab_size, pad_id, device):
        super().__init__()
        embed_table = nn.Embedding(vocab_size, d_model)
        self.encoder = EncoderRNN(embed_table, d_model)
        self.decoder = DecoderRNN(embed_table, d_model, vocab_size)
        self.loss_fct = nn.CrossEntropyLoss()
        self.pad_id = pad_id
        self.device = device

    def forward(self, seq, if_train=True):
        hidden = self.encoder(seq)
        if if_train:
            logits = self.decoder(seq, hidden)
            labels = torch.where(seq["input_ids"] == self.pad_id, torch.tensor(-100).to(self.device), seq["input_ids"])

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            return loss
        else:
            return hidden.squeeze(1)    # N 1 C → N C