import argparse
import torch
import torch.nn as nn
import torch.onnx
import datamy
import model
from torch.utils.data import DataLoader
import torch.optim as optim
import csv
import matplotlib.pyplot as plt
import numpy as np




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

corpus = datamy.Corpus(args.data)
corpus_val = datamy.Corpus_val(args.data)

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

data_loader_en = DataLoader(corpus.english[0:120], batch_size=n_batch, shuffle=False, collate_fn=collate_fn_padd)
data_loader_ha = DataLoader(corpus.hausa[0:120], batch_size=n_batch, shuffle=False, collate_fn=collate_fn_padd)

data_loader_en_val = DataLoader(corpus_val.english_val[0:24], batch_size=n_batch, shuffle=False, collate_fn=collate_fn_padd)
data_loader_ha_val = DataLoader(corpus_val.hausa_val[0:24], batch_size=n_batch, shuffle=False, collate_fn=collate_fn_padd)



n_iteration = len(corpus.english[0:120]) // n_batch
n_iteration_val = len(corpus_val.english_val[0:24]) // n_batch


print("total_iterations: ", n_iteration)
print("total_iterations_val: ", n_iteration_val)


attmodel = model.Attention(src_vocab_size=len(corpus.dictionaryen),
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

    print(lossValid)
    print(lossTrain)



print(lossValid)
print(lossTrain)
plt.plot([step_size * kk for kk in range(1, len(lossValid) + 1)], lossValid, "b", label="validation loss")
plt.plot([step_size * kk for kk in range(1, len(lossTrain) + 1)], lossTrain, "r", label="train loss")
plt.xlabel("number of iteration")
plt.ylabel("losses")
plt.legend()
plt.savefig("ldecay2.png")







"""
src, _= next(iter(data_loader_en))
tar, _= next(iter(data_loader_ha))


hidden_state_init = torch.randn(n_layer, n_batch, 256)
cell_state_init = torch.randn(n_layer, n_batch, 256)

attmodel = model.Attention(src_vocab_size=len(corpus.dictionaryen),
                     tgt_vocab_size=len(corpus.dictionaryha),
                     d_model=256, nhead=8, dim_ffd=1024, dropout1=0, dropout2=0, batch_size=n_batch)

finaloutput = attmodel(src, tar, hidden_state_init, cell_state_init)

print("FINAL!!!!!!!!!!!", finaloutput.shape)
print(tar.shape)

criterion = nn.NLLLoss()

finaloutput = finaloutput.view(-1,11421)
tar = tar.view(-1)

print("FINAL!!!!!!!!!!!", finaloutput.shape)
print(tar.shape)

loss = criterion(finaloutput, tar)
"""






