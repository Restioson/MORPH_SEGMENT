import re

for (lang_name, lang_id) in (("nr", "NR"),):
    with open(f"../Data/{lang_id}_TEST.tsv") as gold_file:
        gold = [line.split("\t") for line in gold_file.read().splitlines(keepends=False) if line.strip()]

    with open(f"model/results.txt") as pred_file:
        pred = [line.split('\t') for line in pred_file.read().splitlines(keepends=False) if line.strip()]

    with open(f"{lang_id}_TEST_CANONICAL_PRED.tsv", "w") as out_file:
        for gold_elt, pred_elt in zip(gold, pred):
            assert gold_elt[2].lower().replace("_", "-") == pred_elt[0]
            pred_seg = re.sub(r"(?<=[a-zA-Z0-9()'])-(?=[a-zA-Z0-9()'])", '_', pred_elt[1])
            out_file.write("\t".join([gold_elt[0], "_", pred_seg, gold_elt[3]]) + "\n")
