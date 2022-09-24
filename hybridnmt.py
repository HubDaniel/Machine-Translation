import os
from io import open
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torch
import torch.nn as nn
import torch.onnx
from torch.utils.data import DataLoader
import torch.optim as optim
import csv
import matplotlib.pyplot as plt
import numpy as np


class Corpus(object):
    def __init__(self, path):
        self.dictionaryen = self.readjason(os.path.join(path, 'en.BPE.txt.json'))
        self.dictionaryha = self.readjason(os.path.join(path, 'ha.BPE.txt.json'))
        self.english = self.TtoIen(os.path.join(path, 'en.BPE.txt'))
        self.hausa = self.TtoIha(os.path.join(path, 'ha.BPE.txt'))

    def readjason(self, path):
        with open(path, encoding='utf-8') as f:
            vocab = json.load(f)
        return vocab

    def TtoIen(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # convert word to indices
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<EOS>']
                ids = []
                for word in words:
                    if word not in self.dictionaryen:
                        ids.append(2)
                    else:
                        ids.append(self.dictionaryen[word])

                idss.append(torch.tensor(ids).type(torch.int64))


        return idss


    def TtoIha(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # convert word to indices
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<EOS>']
                ids = []
                for word in words:
                    if word not in self.dictionaryha:
                        ids.append(2)
                    else:
                        ids.append(self.dictionaryha[word])

                idss.append(torch.tensor(ids).type(torch.int64))


        return idss





class Corpus_val(object):
    def __init__(self, path):
        self.dictionaryen_val = self.readjason(os.path.join(path, 'en_val.BPE.txt.json'))
        self.dictionaryha_val = self.readjason(os.path.join(path, 'ha_val.BPE.txt.json'))
        self.english_val = self.TtoIen(os.path.join(path, 'en_val.BPE.txt'))
        self.hausa_val = self.TtoIha(os.path.join(path, 'ha_val.BPE.txt'))

    def readjason(self, path):
        with open(path, encoding='utf-8') as f:
            vocab = json.load(f)
        return vocab

    def TtoIen(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # convert word to indices
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<EOS>']
                ids = []
                for word in words:
                    if word not in self.dictionaryen_val:
                        ids.append(2)
                    else:
                        ids.append(self.dictionaryen_val[word])

                idss.append(torch.tensor(ids).type(torch.int64))


        return idss


    def TtoIha(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # convert word to indices
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<EOS>']
                ids = []
                for word in words:
                    if word not in self.dictionaryha_val:
                        ids.append(2)
                    else:
                        ids.append(self.dictionaryha_val[word])

                idss.append(torch.tensor(ids).type(torch.int64))


        return idss



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


'''
# load data and store English and Hausa in two list
enlist, halist, info = [], [], []
nbad = 0

tsv_file = open("opus.ha-en.tsv",encoding="utf8")
read_tsv = csv.reader(tsv_file, delimiter="\t")

for n, row in enumerate(read_tsv):
    if len(row) < 3:
        nbad += 1
    else:
        enlist.append(row[0])
        halist.append(row[1])
        info.append(row[2])

tsv_file.close()



# write English and Hausa into two different txt
csv_file = open("en.txt", "w",encoding="utf8")
for i in enlist:
    csv_file.write(i + "\n")
csv_file.close()

csv_file = open("ha.txt", "w",encoding="utf8")
for i in halist:
    csv_file.write(i + "\n")
csv_file.close()

'''




'''
##########################

# In terminal, run the following to create vocabulary
subword-nmt learn-joint-bpe-and-vocab --input en.txt ha.txt -s 10000 -o middle_file.txt --write-vocabulary en2.txt ha2.txt
subword-nmt apply-bpe -c middle_file.txt --vocabulary en2.txt --vocabulary-threshold 50 -i en.txt -o en.BPE.txt
subword-nmt apply-bpe -c middle_file.txt --vocabulary ha2.txt --vocabulary-threshold 50 -i ha.txt -o ha.BPE.txt
python build_dictionary.py en.BPE.txt ha.BPE.txt

##########################
'''









# driving code
parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
parser.add_argument('--data', type=str, default='C:\\Users\\x_r_m\\Desktop\\Learn\\6-2021 Fall\\CS-291K\\HW2_new',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='Transformer',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--emsize', type=int, default=256,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1024,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')

parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')

parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
parser.add_argument('--nhead', type=int, default=8,
                    help='the number of heads in the encoder/decoder of the transformer model')
parser.add_argument('--dry-run', action='store_true',
                    help='verify the code and the model')


parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--dropout1', type=float, default=0,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropout2', type=float, default=0,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--wdecay', type=float, default=0.1,
                    help='decay rate')




args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print("device!!!", device)
###############################################################################
# Load data
###############################################################################

corpus = Corpus(args.data)
corpus_val = Corpus_val(args.data)

src_val = corpus_val.english_val
tar_val = corpus_val.hausa_val

n_batch = args.batch_size
n_layer = args.nlayers


def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    lengths = list(map(len, batch))
    batch = torch.nn.utils.rnn.pad_sequence(batch)
    return batch, lengths

data_loader_en = DataLoader(corpus.english, batch_size=n_batch, shuffle=False, collate_fn=collate_fn_padd)
data_loader_ha = DataLoader(corpus.hausa, batch_size=n_batch, shuffle=False, collate_fn=collate_fn_padd)

data_loader_en_val = DataLoader(corpus_val.english_val, batch_size=n_batch, shuffle=False, collate_fn=collate_fn_padd)
data_loader_ha_val = DataLoader(corpus_val.hausa_val, batch_size=n_batch, shuffle=False, collate_fn=collate_fn_padd)



n_iteration = len(corpus.english) // n_batch - 1
n_iteration_val = len(corpus_val.english_val) // n_batch - 1


print("total_iterations: ", n_iteration)
print("total_iterations_val: ", n_iteration_val)


attmodel = Attention(src_vocab_size=len(corpus.dictionaryen),
                         tgt_vocab_size=len(corpus.dictionaryha),
                         d_model=args.emsize, nhead=args.nhead, dim_ffd=args.nhid, dropout1=args.dropout1, dropout2=args.dropout2, batch_size=n_batch).to(device)
criterion = nn.NLLLoss()
optimizer = optim.Adam(attmodel.parameters(), weight_decay=args.wdecay, lr=args.lr)


data_loader_en_iter = iter(data_loader_en)
data_loader_ha_iter = iter(data_loader_ha)

hidden_state_init = torch.randn(n_layer, n_batch, 256).to(device)
cell_state_init = torch.randn(n_layer, n_batch, 256).to(device)

lossTrain = []
lossValid = []
midlossTrain = 0
step_size = 10

for i in range(1, n_iteration+1):
    src, _ = next(data_loader_en_iter)
    tar, _ = next(data_loader_ha_iter)
    #print(src.shape)
    #print(tar.shape)
    src = src.to(device)
    tar = tar.to(device)
    optimizer.zero_grad()
    finaloutput = attmodel(src, tar, hidden_state_init, cell_state_init)
    finaloutput = finaloutput.view(-1, 11421)
    tar = tar.view(-1)
    loss = criterion(finaloutput, tar)
    # create temperary loss1 for plot
    loss1 = loss
    loss1 = loss1.to('cpu')#.detach().numpy()
    loss1 = loss1.to('cpu').detach()
    midlossTrain += loss1
    print('iteration:', '%04d' % (i), 'cost =', '{:.6f}'.format(loss))
    loss.backward()
    optimizer.step()
    data_loader_en_val1 = data_loader_en_val
    data_loader_en_val_iter = iter(data_loader_en_val1)
    data_loader_ha_val1 = data_loader_ha_val
    data_loader_ha_val_iter = iter(data_loader_ha_val1)



    if i % step_size == 0:
        lossTrain.append(midlossTrain/step_size)
        midlossTrain = 0
        print("validating...")
        total_val_los = 0
        for j in range(n_iteration_val):
            src_val, _ = next(data_loader_en_val_iter)
            tar_val, _ = next(data_loader_ha_val_iter)
            #print(src_val.shape)
            #print(tar_val.shape)
            src_val = src_val.to(device)
            tar_val = tar_val.to(device)
            finaloutput_val = attmodel(src_val, tar_val, hidden_state_init, cell_state_init)
            loss_val = criterion(finaloutput_val.transpose(1, 2), tar_val)
            total_val_los += loss_val
            print('val_iter:', '%04d' % (j + 1), 'val_cost =', '{:.6f}'.format(loss_val))

        total_val_los1 = total_val_los
        total_val_los1 = total_val_los1.to('cpu').detach()
        lossValid.append(total_val_los1/n_iteration_val)

    #print(lossValid)
    #print(lossTrain)



#print(lossValid)
#print(lossTrain)
plt.plot([step_size * kk for kk in range(1, len(lossValid) + 1)], lossValid, "b", label="validation loss")
plt.plot([step_size * kk for kk in range(1, len(lossTrain) + 1)], lossTrain, "r", label="train loss")
plt.xlabel("number of iteration")
plt.ylabel("losses")
plt.legend()
plt.savefig("ldecay2.png")