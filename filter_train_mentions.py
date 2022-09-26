import json
from itertools import cycle

with open("resplited_mentions/train.json", 'r') as f:
    test_mentions = json.load(f)

with open("resplited_tfidfs/train_tfidfs.json", 'r') as f:
    test_tfidfs = json.load(f)

new_mentions = []
new_mention2tf_idfs = []
num_empty_candidates = 0
valid_nums = []
num_candidates_stat = []

print("original number of mentions", len(test_mentions))
for i, mention in enumerate(test_mentions):
    label_id = mention["label_document_id"]
    mention_id = mention["mention_id"]
    for mention2tfidf in iter(test_tfidfs[i:]+test_tfidfs[:i]):
        if mention2tfidf["mention_id"] == mention_id:
            num_candidates_stat.append(len(mention2tfidf["tfidf_candidates"]))
            if not mention2tfidf["tfidf_candidates"]:
                num_empty_candidates += 1
                break
            elif label_id in mention2tfidf["tfidf_candidates"]:
                if len(mention2tfidf["tfidf_candidates"]) < 64: # rm this entry if the num of candidates is less than 64
                    break
                else:
                    cur_label = mention2tfidf["tfidf_candidates"].index(label_id)
            else:
                mention2tfidf["tfidf_candidates"].insert(0, label_id)
                cur_label = 0
                if len(mention2tfidf["tfidf_candidates"]) < 64:
                    # while len(mention2tfidf["tfidf_candidates"]) < 64:
                    #     mention2tfidf["tfidf_candidates"].extend(mention2tfidf["tfidf_candidates"])     # copy to 64
                    #
                    # mention2tfidf["tfidf_candidates"] = mention2tfidf["tfidf_candidates"][:64]
                    break
                else:

                    mention2tfidf["tfidf_candidates"] = mention2tfidf["tfidf_candidates"][:64]

            mention["label"] = cur_label
            valid_nums.append(mention_id)
            new_mentions.append(mention)
            new_mention2tf_idfs.append(mention2tfidf)

print("number of filtered mentions:", len(valid_nums))
assert len(valid_nums) == len(new_mentions)

with open("filtered_mentions/train.json", "w") as f:
    json.dump(new_mentions, f)

with open("filtered_tfidfs/train_tfidfs.json", "w") as f:
    json.dump(new_mention2tf_idfs, f)

with open("train_valid_mentions.json", "w") as f:
    json.dump(valid_nums, f)

import matplotlib.pyplot as plt


plt.figure()
plt.hist(num_candidates_stat, bins=list(range(-1, 65)), )
plt.savefig("train_candidate_stat.png", dpi=300)
plt.show()