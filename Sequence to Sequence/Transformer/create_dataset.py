import json
import random

for (lang_name, lang_id) in (("xhosa", "XH"), ("zulu", "ZU"), ("ndebele", "NR"), ("swati", "SS")):
    for dset in ("train", "test"):
        with open(f"../Data/{lang_id}_{dset.upper()}.tsv") as in_file:
            lines = in_file.read().splitlines(keepends=False)

            if dset == "test":
                with open(f"../Data/{lang_name}/json/{lang_name}-{dset}.json", "w") as out_file:
                    for line in lines:
                        raw, _, segmented, _ = line.split("\t")
                        segmented = segmented.split("_")
                        out_file.write(json.dumps({"src": raw, "trg": '-'.join(segmented)}) + '\n')
            else:
                with open(f"../Data/{lang_name}/json/{lang_name}-valid.json", "w") as valid_file:
                    with open(f"../Data/{lang_name}/json/{lang_name}-train.json", "w") as train_file:
                        random.shuffle(lines)
                        for i, line in enumerate(lines):
                            raw, _, segmented, _ = line.split("\t")
                            segmented = segmented.split("_")

                            out_file = valid_file if i < len(lines) // 10 else train_file
                            out_file.write(json.dumps({"src": raw, "trg": '-'.join(segmented)}) + '\n')
