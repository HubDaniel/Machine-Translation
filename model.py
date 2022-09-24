import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # 5000*1*256
        self.register_buffer('pe', pe)

    def forward(self, x):
        # positional embedding for each sentence in current batch

        x = x.transpose(0,1)
        for i in range(x.shape[0]):
            curSent = x[i, :, :]
            curSent = torch.unsqueeze(curSent, 1)
            curSent = curSent + self.pe[:curSent.size(0), :]
            curSent = torch.squeeze(curSent)
            x[i, :, :] = curSent

        x = x.transpose(0,1)
        #print("after pos_embedding !!!!", x.shape)
        return self.dropout(x)



class Encoder(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, src_vocab_size, d_model, nhead, dim_ffd, nlayers=1, dropout=0):
        super(Encoder, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.encoder = nn.Embedding(src_vocab_size, d_model).to("cuda")
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_ffd, dropout)
        self.transformer_encoder1 = TransformerEncoder(encoder_layers, nlayers)
        self.transformer_encoder2 = TransformerEncoder(encoder_layers, nlayers)


    def forward(self, src):
        src = self.encoder(src)
        src = self.pos_encoder(src)
        output1 = self.transformer_encoder1(src)
        output2 = self.transformer_encoder2(output1)
        return output1, output2


class Attention(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, nhead, dim_ffd, dropout1, dropout2, batch_size):
        super(Attention, self).__init__()
        self.enc_cell = Encoder(src_vocab_size, d_model, nhead, dim_ffd, dropout=dropout1)
        self.dec_cell1 = nn.LSTM(input_size=d_model, hidden_size=d_model, dropout=dropout2)
        self.dec_cell2 = nn.LSTM(input_size=2*d_model, hidden_size=d_model, dropout=dropout2)
        self.embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_embedding = PositionalEncoding(d_model)
        # Linear for attention
        self.attn = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model * 2, tgt_vocab_size)
        self.batch_size = batch_size
        self.d_model = d_model
        self.tgt_vocab_size = tgt_vocab_size

    def forward(self, src, tgt, hidden, cell):

        dec_inputs = self.embedding(tgt)
        dec_inputs = self.pos_embedding(dec_inputs)

        enc_output1, enc_output2 = self.enc_cell(src)
        enc_output2 = enc_output2.to('cuda')

        h = hidden
        c = cell
        n_step = len(dec_inputs)
        layer1 = torch.empty([n_step, self.batch_size, self.d_model * 2]).to('cuda')
        layer2 = torch.empty([n_step, self.batch_size, self.tgt_vocab_size]).to('cuda')



        for i in range(n_step):  # each time step

            dec_output, (h, c) = self.dec_cell1(dec_inputs[i].unsqueeze(0), (h, c))   #[1, batch_size, ]  h [1, batch_size, ] enc [step_e, batch_size, hidden]

            attn_weights = self.get_att_weight(h, enc_output1)
            attn_weights = attn_weights.to('cuda')

            # matrix-matrix product of matrices
            context = attn_weights.bmm(enc_output1.transpose(0, 1))
            dec_output = dec_output.squeeze(0)
            context = context.squeeze(1)
            layer1[i] = torch.cat((dec_output, context), 1)


        h = hidden
        c = cell

        for i in range(n_step):  # each time step
            dec_output, (h, c) = self.dec_cell2(layer1[i].unsqueeze(0), (h, c))
            attn_weights = self.get_att_weight(h, enc_output2)
            attn_weights = attn_weights.to('cuda')
            # matrix-matrix product of matrices [batch_size, 1, enc_length]x[batch_size, enc_length, d_model]=[batch_size, 1, d_model]
            context = attn_weights.bmm(enc_output2.transpose(0, 1))
            dec_output = dec_output.squeeze(0)
            context = context.squeeze(1)
            layer2[i] = self.out(torch.cat((dec_output, context), 1))
        output = F.log_softmax(layer2, dim=2)
        return output

    def get_att_weight(self, h, enc_output):
        n_step = len(enc_output)
        attn_scores = torch.zeros(n_step, self.batch_size)

        for i in range(n_step):
            attn_scores[i] = self.get_att_score(h, enc_output[i])
        attn_scores.transpose(0, 1)
        # Normalize scores to weights in range 0 to 1
        return F.softmax(attn_scores, dim=0).view(self.batch_size, 1, n_step)
        #return F.softmax(attn_scores).view(self.batch_size, 1, -1)

    def get_att_score(self, h, enc_output):  # enc_outputs [batch_size, num_directions(=1) * n_hidden]
        score = self.attn(enc_output)  # score : [batch_size, n_hidden]
        h = torch.squeeze(h, 0)
        att_score = torch.zeros(self.batch_size)
        for i in range(self.batch_size):
            att_score[i] = torch.dot(h[i].view(-1), score[i].view(-1))
        return att_score # inner product make scalar value