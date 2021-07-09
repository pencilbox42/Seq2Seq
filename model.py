import math
import torch
import random
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self,
                 input_size,
                 embed_size,
                 hidden_size,
                 num_layers,
                 dropout):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size

        self.embedding = nn.Embedding(self.input_size, self.embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers, dropout=dropout, bidirectional=True)

    def forward(self, src, hidden=None):
        embedded = self.embed(src)
        outputs, hidden = self.rnn(embedded, hidden)
        outputs = (outputs[:,:,:self.hidden_size] + outputs[:,:,:self.hidden_size])
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self,hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size*2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1./math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        time_step = encoder_outputs.size(0)
        h = hidden.repeat(time_step, 1, 1).transpose(0,1)
        encoder_outputs = encoder_outputs.transpose(0,1)
        attn_energy = self.score(h, encoder_outputs)
        attention = F.softmax(attn_energy, dim=1).unsqueeze(1)
        return attention

    def score(self, hidden, encoder_outputs):
        energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1,2)  # [B, T, H] -> [B, H, T]
        v = self.v.repeat(encoder_outputs.size(0),1).unsqueeze(1)  #[B, 1, H]
        energy = torch.bmm(v, energy)  #[B, 1, H] ** [B, H, T] -> [B, 1, T]
        return energy

class Decoder(nn.Module):
    def __init__(self,
                 embed_size,
                 hidden_size,
                 output_size,
                 num_layers,
                 dropout):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size)
        self.rnn = nn.GRU(hidden_size+embed_size, hidden_size, num_layers, dropout=dropout)
        self.output = nn.Linear(hidden_size*2, output_size)

    def forward(self, input, last_hidden, encoder_outputs):
        embedded = self.embedding(input).unsqueeze(0)  #[1, B, N]
        embedded = self.dropout(embedded)
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0,1))  #[B, 1, N]
        context = context.transpose(0,1)  #[1, B, N]
        rnn_input = torch.cat([context, embedded], dim=2)

        output, hidden = self.rnn(rnn_input, last_hidden)

        output = output.squeeze(0)  #[1, B, N] -> [B, N]
        context = context.squeeze(0)  #[1, B, N] -> [B, N]
        output = self.out(torch.cat([output, context], dim=1))
        output = F.log_softmax(output, dim=1)
        return output, hidden, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing=0.5):
        batch_size = src.size(1)
        max_len = trg.size(0)
        vocab_size = self.decoder.output_size
        outputs = Variable(torch.zeros(max_len, batch_size, vocab_size)).cuda()

        encoder_output, hidden = self.encoder(src)
        hidden = hidden[:self.decoder.num_layers]
        output = Variable(trg.data[0, :])  # sos
        for t in range(1, max_len):
            output, hidden, attn_weights = self.decoder(output, hidden, encoder_output)
            outputs[t] = output
            is_teacher = random.random() < teacher_forcing
            top1 = output.data.max(1)[1]
            output = Variable(trg.data[t] if is_teacher else top1).cuda()
        return outputs


