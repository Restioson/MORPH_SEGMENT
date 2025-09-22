import torch
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from aligned_f1 import align_seqs


def evaluate_model(model, iterator, criterion):
    """Function to evaluate the model inbetween each epoch"""

    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output, _ = model(src, trg[:, :-1])
            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50):
    model.eval()
    tokens = [token.lower() for token in sentence]
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        pred_token = output.argmax(2)[:, -1].item()

        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:], attention


def _beam_search_inner(prev_prob, prev_target_tokens, src_encoded, src_mask, model, device, best_k, min_prob_to_keep):
    target_tensor = torch.LongTensor(prev_target_tokens).unsqueeze(0).to(device)
    target_mask = model.make_trg_mask(target_tensor)

    with torch.no_grad():
        output, attention = model.decoder(target_tensor, src_encoded, target_mask, src_mask)

    probabilities = torch.nn.functional.softmax(output, dim=2)[:, -1].flatten()
    top_k = torch.topk(probabilities, k=best_k)

    possibilities = []
    for i in range(best_k):
        token, prob = top_k.indices[i].item(), top_k.values[i].item()
        if prob < min_prob_to_keep:
            continue

        possibilities.append((prev_prob + prob, prev_target_tokens + [token]))

    return possibilities


def _convert_indices_to_morphemes(target_indices, target_field):
    joined = "".join([target_field.vocab.itos[i] for i in target_indices[1:]])
    joined = joined.replace("<eos>", "").split("-")
    return joined


def beam_search_word(word, src_field, target_field, model, device, max_len=100, best_k=5, min_prob_to_keep=0.15):
    model.eval()
    src_tokens = [char.lower() for char in word]
    src_tokens = [src_field.init_token] + src_tokens + [src_field.eos_token]

    src_indexes = [src_field.vocab.stoi[token] for token in src_tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        src_encoded = model.encoder(src_tensor, src_mask)

    start_target_tokens = [target_field.vocab.stoi[target_field.init_token]]

    final_segmentations = []
    in_progress_branches = [(0.0, start_target_tokens)]
    while len(in_progress_branches) != 0:
        # Predict the next token for each branch under consideration
        new_branches = []
        for (prob, branch) in in_progress_branches:
            new_branch = _beam_search_inner(prob, branch, src_encoded, src_mask, model, device, best_k,
                                            min_prob_to_keep)
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
        for (prob, branch) in all_branches[:best_k]:
            if len(branch) == max_len or branch[-1] == target_field.vocab.stoi[target_field.eos_token]:
                final_segmentations.append((prob, branch))
            else:
                in_progress_branches.append((prob, branch))

    trg_tokens = sorted(
        ((prob, _convert_indices_to_morphemes(target_indices, target_field))
         for (prob, target_indices) in final_segmentations),
        key=lambda pair: pair[0]
    )
    return list(trg_tokens)


def display_attention(sentence, translation, attention, n_heads=8, n_rows=4, n_cols=2):
    """Function to display attention over the segmented word"""

    assert n_rows * n_cols == n_heads

    fig = plt.figure(figsize=(15, 25))

    for i in range(n_heads):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)

        _attention = attention.squeeze(0)[i].cpu().detach().numpy()

        ax.matshow(_attention)

        ax.tick_params(labelsize=12)
        ax.set_xticklabels([''] + ['<sos>'] + [t.lower() for t in sentence] + ['<eos>'],
                           rotation=45)
        ax.set_yticklabels([''] + translation)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()


def array_to_word(translation):
    out = ""
    for letter in translation:
        out += letter
    return out


def evaluate_words(n, test_data, src_field, target_field, model, device, print_words=True, save=False):
    """Function to evaluate words in the test set"""

    if n == -1:
        n = len(test_data.examples)

    file = open('results.txt', 'w')
    for i in range(n):
        src = vars(test_data.examples[i])['src']
        trg = vars(test_data.examples[i])['trg']

        translation, attention = translate_sentence(src, src_field, target_field, model, device)

        if print_words:
            print("Target:\t", array_to_word(trg))
            print("Prediction:\t", array_to_word(translation))
            print("")

        if save:
            file.write(array_to_word(trg) + '\t' + array_to_word(translation).replace("<eos>", "") + '\n')
    file.close()


def repl(src_field, target_field, model, device):
    while True:
        src = input("> ")
        for word in src.split(" "):
            translation, attention = translate_sentence(word, src_field, target_field, model, device)
            print(array_to_word(translation).replace("<eos>", ""))


def segment_one(src_field, target_field, model, device, text):
    results = []
    for word in text.split(" "):
        translation, attention = translate_sentence(word, src_field, target_field, model, device)
        results.append(array_to_word(translation).replace("<eos>", "").split("-"))
    return results


def f1_scores(valid, src_field, target_field, model, device, align=False):
    pad = '__<PAD>__'

    correct = 0
    model.eval()
    with torch.no_grad():
        all_targets, all_preds = [], []
        for ex in valid.examples:
            src = ex.src
            trg = ex.trg

            translation, _ = translate_sentence(src, src_field, target_field, model, device)
            target = array_to_word(trg).split("-")
            prediction = array_to_word(translation).replace("<eos>", "").split("-")
            assert pad not in prediction and pad not in target

            if align:
                prediction, target = align_seqs(prediction, target)
            else:
                desired_len = max(len(prediction), len(target))
                prediction += [pad] * (desired_len - len(prediction))
                target += [pad] * (desired_len - len(target))

                if target == prediction:
                    correct += 1

            all_targets.extend(target)
            all_preds.extend(prediction)
        print("done examples")
        micro = f1_score(all_targets, all_preds, zero_division=0.0, average='micro')
        macro = f1_score(all_targets, all_preds, zero_division=0.0, average='macro')
        accuracy = correct / len(valid.examples)
    return micro, macro, accuracy
