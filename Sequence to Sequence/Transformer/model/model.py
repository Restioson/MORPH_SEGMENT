
import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from aligned_f1 import align_seqs


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys
import numpy as np

import random
import math
import time
import data
import evaluate
SEED = 1234

sys.path.append("/home/restioson/PycharmProjects/MORPH_PARSE/from_scratch")

from demo import load_model, eval_segmented

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LENGTH = 1000


'''
    Encoder block made up of the encoder layers: positional embedding, multi-head attention, feed foward and dropout
'''
class Encoder(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim,
                 dropout, 
                 device,
                 max_length = MAX_LENGTH):
        super().__init__()

        self.device = device
        
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim,
                                                  dropout, 
                                                  device) 
                                     for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, src, src_mask):
        
        
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        
        
        for layer in self.layers:
            src = layer(src, src_mask)
            
            
        return src

'''
    Encoder layer to make up encoder block, specifies the hidden dimension, number of heads etc..
'''
class EncoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim,  
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        
        _src, _ = self.self_attention(src, src, src, src_mask)
        
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        
        
        _src = self.positionwise_feedforward(src)
        
        src = self.ff_layer_norm(src + self.dropout(_src))
        
        
        return src


'''
    Multi-Head Attention block, fits into the encoder and decoder
'''
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        
        batch_size = query.shape[0]
        
                
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
                
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
                
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim = -1)
                
                
        x = torch.matmul(self.dropout(attention), V)
        
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        
        x = self.fc_o(x)
        
        
        return x, attention

'''
    Feed Forward network to fit into encoder and decoder blocks
'''
class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        
        x = self.dropout(torch.relu(self.fc_1(x)))
        
        
        x = self.fc_2(x)
        
        
        return x
'''
    Decoder block made up of the decoder layers: positional embedding, (masked) multi-head attention, feed foward and dropout
'''
class Decoder(nn.Module):
    def __init__(self, 
                 output_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device,
                 max_length = MAX_LENGTH):
        super().__init__()
        
        self.device = device
        
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim, 
                                                  dropout, 
                                                  device)
                                     for _ in range(n_layers)])
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
                
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
                            
            
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
                
        
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        
        
        output = self.fc_out(trg)
        
            
        return output, attention
'''
    Decoder layer to make up decoder block, specifies the hidden dimension, number of heads etc..
'''
class DecoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        
        
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
            
            
        
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        
        
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
                    
        

        _trg = self.positionwise_feedforward(trg)
        

        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        
        
        return trg, attention


'''
    Entire model tied together with the encoder and decoder
'''
class Seq2Seq(nn.Module):
    def __init__(self, 
                 encoder, 
                 decoder, 
                 src_pad_idx, 
                 trg_pad_idx, 
                 device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        
        
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)


        return src_mask
    
    def make_trg_mask(self, trg):
        
        
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        
        
        trg_len = trg.shape[1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        
            
        trg_mask = trg_pad_mask & trg_sub_mask
        
        
        return trg_mask

    def forward(self, src, trg):
        
                
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        
        enc_src = self.encoder(src, src_mask)
        
                
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        
        
        return output, attention



d = data.data()
train_iterator, valid_iterator, test_iterator, test_data, train_data, valid_data, SRC, TRG = d.getIterators()


'''
    Specify the input, hidden, ouput dimensions. Encoder, Decoder heads and dropout
'''
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
HID_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

enc = Encoder(INPUT_DIM, 
              HID_DIM, 
              ENC_LAYERS, 
              ENC_HEADS, 
              ENC_PF_DIM, 
              ENC_DROPOUT, 
              device)

dec = Decoder(OUTPUT_DIM, 
              HID_DIM, 
              DEC_LAYERS, 
              DEC_HEADS, 
              DEC_PF_DIM, 
              DEC_DROPOUT, 
              device)

SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

'''
    Initialise model
'''
model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

'''
    Function to compare model size with hyper-parameter changes
'''
# print(f'The model has {count_parameters(model):,} trainable parameters')

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

model.apply(initialize_weights);


'''
    Specify learning rate and optimisation function
'''
LEARNING_RATE = 0.0005

optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)


'''
    Function to train the model
'''
def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()
        
        output, _ = model(src, trg[:,:-1])
                
            
        output_dim = output.shape[-1]
            
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)
                
            
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_EPOCHS = 150
CLIP = 1

best_valid_loss = float('inf')

# # '''
# #     Training loop for N_Epochs
# # '''
# for epoch in range(N_EPOCHS):
#     start_time = time.time()
#     e = evaluate.evaluate(model)
#     train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
#     valid_loss = e.evaluateModel(model, valid_iterator, criterion)
#     print("Valid Loss")
#     end_time = time.time()
#     epoch_mins, epoch_secs = epoch_time(start_time, end_time)
#
#     if valid_loss < best_valid_loss:
#         best_valid_loss = valid_loss
#         torch.save(model.state_dict(), 'segment.pt')
#
#     print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
#     print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
#     print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
#
#     # if epoch % 100 == 0 and epoch > 0:
#     #     micro, macro = e.f1_scores(valid_data, SRC, TRG, model, device)
#     #     print(f'Micro F1: {micro:.6f}. Macro F1: {macro:.6f}')


'''
    Load best saved model for evaluation
'''
model.load_state_dict(torch.load('segment_zu.pt', map_location=torch.device('cpu'), weights_only=True))
e = evaluate.evaluate(model)
test_loss = e.evaluateModel(model, test_iterator, criterion)
# e.repl(SRC, TRG, model, device)

# print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
# print(e.f1_scores(test_data, SRC, TRG, model, device))

tagger = load_model("/home/restioson/PycharmProjects/MORPH_PARSE/out_models/bilstm_word_morpheme_canon/bilstm-word-morpheme-ZU.pt")

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    GREY = '\033[90m'

# Imbalwa kakhulu imibutho esebenzayo
# Intliziyo yoMgaqo-siseko nguMthetho Oyilwayo waMalungelo
# Yintoni oyenzayo ekwenza kubenzima ukuthetha

# correct_and_raw = [
#     (
#         # i-mbalwa ka-khulu i-mi-butho e-sebenz-a-yo
#         # i-n-tliziyo ya-u-m-gaqo-si-seko ng-u-m-thetho o-yil-w-a-yo wa-a-ma-lungelo
#         # yi-n-to-ni o-yi-enz-a-yo e-u-ku-enz-a ku-be-nzima u-ku-theth-a
#
#         "Imbalwa kakhulu imibutho esebenzayo",
#         [
#             "i[CopPre]-mbalwa[RelStem]",
#             "ka[AdvPre]-khulu[AdjStem]",
#             "i[NPrePre4]-mi[BPre4]-butho[NStem]",
#             "e[RelConc4]-sebenz[VRoot]-a[VerbTerm]-yo[RelSuf]",
#         ],
#     ),
#     (
#         "Intliziyo yoMgaqo-siseko nguMthetho Oyilwayo waMalungelo",
#             [
#             "i[NPrePre9]-n[BPre9]-tliziyo[NStem]",
#             "ya[PossConc9]-u[NPrePre3]-m[BPre3]-gaqo[NStem]-si[BPre7]-seko[NStem]",
#             "ng[CopPre]-u[NPrePre3]-m[BPre3]-thetho[NStem]",
#             "o[RelConc3]-yil[VRoot]-w[PassExt]-a[VerbTerm]-yo[RelSuf]",
#             "wa[PossConc3]-a[NPrePre6]-ma[BPre6]-lungelo[NStem]",
#         ],
#     ),
#     (
#         "Yintoni oyenzayo ekwenza kubenzima ukuthetha",
#         [
#         "yi[CopPre]-n[BPre9]-to[NStem]-ni[InterrogSuf]",
#         "o[RelConc1]-yi[OC9]-enz[VRoot]-a[VerbTerm]-yo[RelSuf]",
#         "e[LocPre]-u[NPrePre15]-ku[BPre15]-enz[VRoot]-a[VerbTerm]",
#         "ku[CopPre]-be[CopPre]-nzima[RelStem]",
#         "u[NPrePre15]-ku[BPre15]-theth[VRoot]-a[VerbTerm]",
#     ])
# ]
#
# preds = [
#     [
#         "i[SC9]-mbalwa[RelStem]",
#         "ka[AdvPre]-khulu[AdjStem]",
#         "i[NPrePre4]-mi[BPre4]-butho[NStem]",
#         "e[RelConc9]-sebenz[VRoot]-a[VerbTerm]-yo[RelSuf]",
#         ".[Punc]"
#     ],
#     [
#         "i[NPrePre9]-n[BPre9]-tliziyo[NStem]",
#         "ya[PossConc9]-u[NPrePre3]-m[BPre3]-gaqo[NStem]-si[BPre7]-seko[NStem]",
#         "ng[CopPre]-u[NPrePre3]-m[BPre3]-thetho[NStem]",
#         "o[RelConc3]-yil[VRoot]-w[PassExt]-a[VerbTerm]-yo[RelSuf]",
#         "wa[PossConc3]-a[NPrePre6]-ma[BPre6]-lungelo[NStem]",
#         ".[PUNC]",
#     ],
#     [
#         "yi[CopPre]-n[BPre9]-to[NStem]-ni[InterrogSuf]",
#         "o[RelConc1]-yi[OC9]-enz[VRoot]-a[VerbTerm]-yo[RelSuf]",
#         "e[RelConc9]-u[NPrePre15]-ku[BPre15]-enz[VRoot]-a[VerbTerm]",
#         "ku[CopPre]-be[CopPre]-nzima[RelStem]",
#         "u[NPrePre15]-ku[BPre15]-theth[VRoot]-a[VerbTerm]",
#         "?[Punc]"
#     ]
# ]

while True:

# for (raw, correct), pred in zip(correct_and_raw, preds):
    raw = input("> ")
    # time.sleep(0.2)

    # for letter in raw:
    #     print(letter, end="", flush=True)
    #     time.sleep((60.0 / 397.0) / 4 * abs(random.normalvariate(mu=1.0, sigma=0.2)))
    # time.sleep(0.1)

    segmented = e.segment_one(SRC, TRG, model, device, raw)
    pred = eval_segmented(tagger, segmented)

    # for pred_word, correct_word in zip(pred, correct):
    # for word in pred:
        # print(bcolors.ENDC, "Truth:   ", bcolors.GREY, f"{correct_word}")
    # print(bcolors.ENDC, "Predicted: ", end="")
    # print(segmented)
    print(pred)
    # pred_word = pred_word.split("-")
    # correct_word = correct_word.split("-")
    # pred_word, correct_word = align_seqs(pred_word, correct_word, pad="[padded]")
    # for i, (morph_pred, morph_correct) in enumerate(zip(pred_word, correct_word)):
    #     color = bcolors.OKGREEN if morph_pred == morph_correct else bcolors.FAIL
    #     start_hypen = bcolors.GREY + "-" if i > 0 else ""
    #     print(f"{start_hypen}{color}{morph_pred}", end="")
    # print(bcolors.ENDC, "\n")

'''
    Function to evaluate words in the test set
'''
# e.evaluateWords(-1, test_data, SRC, TRG, model, device, save=True, printWords=False)
