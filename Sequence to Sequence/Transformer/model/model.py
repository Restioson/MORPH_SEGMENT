import csv
import re
import sys

import math
import random
import time

import torch
import torch.nn as nn
import numpy as np
import tqdm

import dataset
import evaluation

# Disable inspections since the import comes from the below path and PyCharm does not understand this
sys.path.append("/home/restioson/PycharmProjects/MORPH_PARSE/from_scratch")
# noinspection PyUnresolvedReferences,PyPackageRequirements
from demo import load_model, predict_tags_for_word  # noqa: E402

MAX_LENGTH = 1000


class Encoder(nn.Module):
    """
    Encoder block made up of the encoder layers: positional embedding, multi-head attention, feed foward and dropout
    """

    def __init__(self,
                 input_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length=MAX_LENGTH):
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


class EncoderLayer(nn.Module):
    """Encoder layer to make up encoder block, specifies the hidden dimension, number of heads etc..."""

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


class MultiHeadAttentionLayer(nn.Module):
    """Multi-Head Attention block, fits into the encoder and decoder"""

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

    def forward(self, query, key, value, mask: torch.Tensor = None):
        batch_size = query.shape[0]

        q = self.fc_q(query)
        k = self.fc_k(key)
        v = self.fc_v(value)

        q = q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(q, k.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)

        x = torch.matmul(self.dropout(attention), v)

        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(batch_size, -1, self.hid_dim)

        x = self.fc_o(x)

        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    """Feed Forward network to fit into encoder and decoder blocks"""

    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc_1(x)))

        x = self.fc_2(x)

        return x


class Decoder(nn.Module):
    """
    Decoder block made up of the decoder layers: positional embedding, (masked) multi-head attention, feed foward
    and dropout
    """

    def __init__(self,
                 output_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length=MAX_LENGTH):
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

        tok_embedding = self.tok_embedding(trg)
        pos_embedding = self.pos_embedding(pos)
        trg = self.dropout((tok_embedding * self.scale) + pos_embedding)

        attention = None
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        output = self.fc_out(trg)

        return output, attention


class DecoderLayer(nn.Module):
    """Decoder layer to make up decoder block, specifies the hidden dimension, number of heads etc.."""

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


class Seq2Seq(nn.Module):
    """Entire model tied together with the encoder and decoder"""

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

    def make_src_mask(self, src: torch.Tensor):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg: torch.Tensor):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output, attention


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, iterator, optimizer, criterion, clip):
    """Function to train the model"""
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output, _ = model(src, trg[:, :-1])

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

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


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def train(model, src_field, target_field, train_iter, valid_iter, valid_data, device):
    target_pad_idx = target_field.vocab.stoi[target_field.pad_token]

    model.apply(initialize_weights)

    # Function to compare model size with hyper-parameter changes
    print(f'The model has {count_parameters(model):,} trainable parameters')

    # Specify learning rate and optimisation function
    lr = 0.0005
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=target_pad_idx)
    n_epochs = 150
    gradient_clip = 1

    best_valid_loss = float('inf')

    # Training loop for N_Epochs
    for epoch in range(n_epochs):
        start_time = time.time()
        train_loss = train_epoch(model, train_iter, optimizer, criterion, gradient_clip)
        valid_loss = evaluation.evaluate_model(model, valid_iter, criterion)
        print("Valid Loss")
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'segment.pt')

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    micro, macro = evaluation.f1_scores(valid_data, src_field, target_field, model, device)
    print(f'Micro F1: {micro:.6f}. Macro F1: {macro:.6f}')


class TransformerSegmenterBeamSearch:
    def __init__(
            self,
            data: dataset.Data,
            max_len=50,
            best_k=5,
            min_prob_to_keep=0,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    ):
        self.best_k = best_k
        self.min_prob_to_keep = min_prob_to_keep
        self.max_len = max_len
        self.device = device
        self.src_field, self.trg_field = data.src_field, data.target_field

        input_dim = len(self.src_field.vocab)
        output_dim = len(self.trg_field.vocab)
        hidden_dim = 256
        encoder_layers = 3
        decoder_layers = 3
        encoder_heads = 8
        decoder_heads = 8
        encoder_pf_dim = 512
        decoder_pf_head = 512
        encoder_dropout = 0.1
        decoder_dropout = 0.1

        enc = Encoder(input_dim,
                      hidden_dim,
                      encoder_layers,
                      encoder_heads,
                      encoder_pf_dim,
                      encoder_dropout,
                      device)

        dec = Decoder(output_dim,
                      hidden_dim,
                      decoder_layers,
                      decoder_heads,
                      decoder_pf_head,
                      decoder_dropout,
                      device)

        src_pad_idx = self.src_field.vocab.stoi[self.src_field.pad_token]
        trg_pad_idx = self.trg_field.vocab.stoi[self.trg_field.pad_token]

        self.model = Seq2Seq(enc, dec, src_pad_idx, trg_pad_idx, device).to(device)

    def segment_word(self, word):
        src_tokens = [char.lower() for char in word]
        src_tokens = [self.src_field.init_token] + src_tokens + [self.src_field.eos_token]

        src_indexes = [self.src_field.vocab.stoi[token] for token in src_tokens]
        src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(self.device)
        src_mask = self.model.make_src_mask(src_tensor)

        with torch.no_grad():
            src_encoded = self.model.encoder(src_tensor, src_mask)

        start_target_tokens = [self.trg_field.vocab.stoi[self.trg_field.init_token]]

        final_segmentations = []
        in_progress_branches = [(0.0, start_target_tokens)]
        while len(in_progress_branches) != 0:
            # Predict the next token for each branch under consideration
            new_branches = []
            for (prob, branch) in in_progress_branches:
                new_branch = self._beam_search_inner(prob, branch, src_encoded, src_mask)
                new_branches.extend(new_branch)

            # Take top `best_k` branches and use those as either the final segmentations or new in-progress branches
            all_branches = new_branches + final_segmentations
            all_branches = sorted(
                all_branches,
                reverse=True,
                key=lambda prob_and_branch: prob_and_branch[0] / len(prob_and_branch[1])
            )

            in_progress_branches = []
            final_segmentations = []
            for (prob, branch) in all_branches[:self.best_k]:
                if len(branch) == self.max_len or branch[-1] == self.trg_field.vocab.stoi[self.trg_field.eos_token]:
                    final_segmentations.append((prob, branch))
                else:
                    in_progress_branches.append((prob, branch))

        trg_tokens = sorted(
            ((prob, self._convert_indices_to_morphemes(target_indices))
             for (prob, target_indices) in final_segmentations),
            key=lambda pair: pair[0]
        )
        return list(trg_tokens)

    def _beam_search_inner(self, prev_prob, prev_target_tokens, src_encoded, src_mask):
        target_tensor = torch.LongTensor(prev_target_tokens).unsqueeze(0).to(self.device)
        target_mask = self.model.make_trg_mask(target_tensor)

        with torch.no_grad():
            output, attention = self.model.decoder(target_tensor, src_encoded, target_mask, src_mask)

        probabilities = torch.nn.functional.softmax(output, dim=2)[:, -1].flatten()
        top_k = torch.topk(probabilities, k=self.best_k)

        possibilities = []
        for i in range(self.best_k):
            token, prob = top_k.indices[i].item(), top_k.values[i].item()
            if prob < self.min_prob_to_keep:
                continue

            possibilities.append((prev_prob + prob, prev_target_tokens + [token]))

        return possibilities

    def _convert_indices_to_morphemes(self, target_indices):
        joined = "".join([self.trg_field.vocab.itos[i] for i in target_indices[1:]])
        joined = joined.replace("<eos>", "").split("-")
        return joined

    def load_state_dict(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))


def beam_search():
    data = dataset.Data()
    model = TransformerSegmenterBeamSearch(data)
    model.load_state_dict(
        torch.load(
            'segment_new_zu_no_validset.pt',
            map_location=torch.device('cpu'),
            weights_only=True
        )
    )

    tagger = load_model(
        "/home/restioson/PycharmProjects/MORPH_PARSE/bilstm-no-testset-words-morpheme-ZU.pt"
    )

    for best_k in [1, 2, 3, 4, 5]:
        min_prob_to_keep = 0.00
        max_len = 50

        print(f"\n\nk = {best_k}")

        tags_correct = 0
        seg_correct = 0
        analysis_correct = 0
        gold_tag_hr = 0
        total_segs = []
        incorrect = dict()
        for i, example in enumerate(valid_data):
            word = "".join(example.src)
            true_morphemes = "".join(example.trg).lower().split("-")
            true_tags = example.tags.split("_")

            pred_morphemes = model.segment_word(word)
            pred_analyses = []
            pred_tags = []
            for possibility in pred_morphemes:
                tags = predict_tags_for_word(tagger, possibility)
                pred_analyses.append((possibility, tags))
                pred_tags.append(tags)

            if true_tags == predict_tags_for_word(tagger, true_morphemes):
                gold_tag_hr += 1

            if (true_morphemes, true_tags) in pred_analyses:
                # print(word, "correct")
                analysis_correct += 1
            else:
                if true_morphemes in pred_morphemes:
                    seg_correct += 1

                if true_tags in pred_tags:
                    tags_correct += 1

                # print(word, "incorrect")
                incorrect[i] = {"word": word, "true": (true_morphemes, true_tags), "tried": pred_analyses}

            total_segs.append(len(pred_morphemes))

        # print("All incorrect examples")
        # pprint.pprint(incorrect)

        print(f"Gold tag hitrate = {gold_tag_hr / len(valid_data) * 100:.2f}")
        print(f"{analysis_correct / len(valid_data) * 100:.2f}% analysis lattice coverage (seg + parse)")
        incorrect_with_punc = [i for i in incorrect.values() if not i["word"].isalpha()]
        print(f"{len(incorrect_with_punc)} incorrect examples with numbers or punctuation out of"
              f" {len(incorrect)} errors total")
        print(f"If these errors were fixed, then analysis lattice coverage would be "
              f"{(analysis_correct + len(incorrect_with_punc)) / len(valid_data) * 100:.2f}")
        print(
            f"{(seg_correct + analysis_correct) / len(valid_data) * 100:.2f}% segmentation lattice coverage (seg only)"
        )
        print(f"{(tags_correct + analysis_correct) / len(valid_data) * 100:.2f}% tag lattice coverage (tag only)")

        acc = 78.25  # Cached manually ;)
        print("Accuracy on the greedy best choice of segmentation", acc)


def segment_and_tag_unseen():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    d = dataset.Data()
    train_data, train_iter, valid_data, valid_iter, test_data, test_iter, src_field, target_field = d.get_iterators()

    # Specify the input, hidden, ouput dimensions. Encoder, Decoder heads and dropout
    input_dim = len(src_field.vocab)
    output_dim = len(target_field.vocab)
    hidden_dim = 256
    encoder_layers = 3
    decoder_layers = 3
    encoder_heads = 8
    decoder_heads = 8
    encoder_pf_dim = 512
    decoder_pf_head = 512
    encoder_dropout = 0.1
    decoder_dropout = 0.1

    enc = Encoder(input_dim,
                  hidden_dim,
                  encoder_layers,
                  encoder_heads,
                  encoder_pf_dim,
                  encoder_dropout,
                  device)

    dec = Decoder(output_dim,
                  hidden_dim,
                  decoder_layers,
                  decoder_heads,
                  decoder_pf_head,
                  decoder_dropout,
                  device)

    SRC_PAD_IDX = src_field.vocab.stoi[src_field.pad_token]
    TRG_PAD_IDX = target_field.vocab.stoi[target_field.pad_token]

    # Initialise model
    model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

    # # Load best saved model for evaluation
    model.load_state_dict(
        torch.load(
            'segment_new_zu_no_validset.pt',
            map_location=torch.device('cpu'),
            weights_only=True
        )
    )

    tagger = load_model(
        "/home/restioson/PycharmProjects/MORPH_PARSE/bilstm-no-testset-words-morpheme-ZU.pt"
    )

    unseen = []
    with open("Corpus_Zulu.txt", "rb") as f:
        for line_no, line in enumerate(f.read().decode("utf-8", errors="ignore").split("\n")):
            unseen.extend((line_no + 1, word) for word in line.split())

    with open("corpus_analysed.tsv", "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            ["Raw word", "Line in input", "Morphological analysis", "Morphemes", "Morphological tags"]
        )

        for (line_no, word) in tqdm.tqdm(unseen):
            morphemes = evaluation.segment_one(src_field, target_field, model, device, word)[0]
            tags = predict_tags_for_word(tagger, morphemes)
            analysis = "-".join([f"{morpheme}[{tag}]" for morpheme, tag in zip(morphemes, tags)])
            writer.writerow([word, str(line_no), analysis, "-".join(morphemes), "-".join(tags)])


def find_class_5_or_11():
    with open("corpus_analysed.tsv", "r") as in_f:
        reader = csv.reader(in_f, delimiter="\t")
        next(reader)

        by_line = []
        cur_line = 0
        for record in reader:
            if (line_no := record[1]) != cur_line:
                by_line.append([])
                cur_line = line_no
            by_line[-1].append(record)

        with open("corpus_filtered_class_11_or_5.tsv", "w") as out_f:
            writer = csv.writer(out_f, delimiter="\t")
            for line in by_line:
                #print(any("11" in (analysis := word[4]) for word in line))
                # print(any(re.match(r"[a-zA-Z]5]", word[4]) for word in line))
                if (any(re.search(r"[a-zA-Z]5(-|$)", word[4]) for word in line)):
                    print("HI")
                #print('='*100)

                if any("11" in word[4] or re.search(r"[a-zA-Z]5(-|$)", word[4]) for word in line):
                    writer.writerow([
                        " ".join(word[0] for word in line),  # Raw words
                        line[0][1],  # Line no
                        " ".join(word[2] for word in line),  # Analysis
                        " ".join(word[3] for word in line),  # Morphemes
                        " ".join(word[4] for word in line),  # Tags
                    ])


if __name__ == "__main__":
    find_class_5_or_11()
