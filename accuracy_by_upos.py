from collections import Counter

import numpy as np
from tqdm import tqdm
from trankit import Pipeline

from data import load_df

if __name__ == "__main__":
	LANG = "slovenian"  # {"croatian", "slovenian"}
	data_paths = [
		"komet-fold0-csebert-2e-5-binary3-history0-optthresh/predict-test/test_results.tsv",
		"komet-fold1-csebert-2e-5-binary3-history0-optthresh/predict-test/test_results.tsv",
		"komet-fold2-csebert-2e-5-binary3-history0-optthresh/predict-test/test_results.tsv",
		"komet-fold3-csebert-2e-5-binary3-history0-optthresh/predict-test/test_results.tsv",
		"komet-fold4-csebert-2e-5-binary3-history0-optthresh/predict-test/test_results.tsv"
	]

	all_words, all_token_preds, all_token_true = [], [], []
	for curr_path in data_paths:
		df = load_df(curr_path)

		all_words.extend(df["sentence_words"].tolist())
		all_token_preds.extend(df["preds_transformed"].tolist())
		all_token_true.extend(df["true_transformed"].tolist())

	print(f"Analyzing {len(all_words)} examples...")
	pipe = Pipeline(LANG, embedding='xlm-roberta-large')

	UPOS_TAGSET = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM",  "VERB", "X"]
	upos_to_correctness = {tag: {"correct": [], "incorrect": []} for tag in UPOS_TAGSET}
	for idx_ex, (words, token_preds, token_true) in tqdm(enumerate(zip(all_words, all_token_preds, all_token_true)),
														 total=len(all_words)):
		res = pipe.posdep(words, is_sent=True)
		token_annotations = res["tokens"]
		assert len(token_annotations) == len(words)
		upostags = list(map(lambda token_info: token_info["upos"], token_annotations))

		for word, upos, word_pred, word_true in zip(words, upostags, token_preds, token_true):
			if word_true in {"O", 0, "not_metaphor"}:
				continue

			if word_pred == word_true:
				existing = upos_to_correctness[upos]["correct"]
				existing.append(word)
				upos_to_correctness[upos]["correct"] = existing
			else:
				existing = upos_to_correctness[upos]["incorrect"]
				existing.append(word)
				upos_to_correctness[upos]["incorrect"] = existing

	correctness = []
	for upos in UPOS_TAGSET:
		upos_stats = upos_to_correctness[upos]
		num_correct = len(upos_stats['correct'])
		num_total = num_correct + len(upos_stats['incorrect'])
		correctness.append(num_correct / max(1, num_total))

	sort_indices = np.argsort(correctness)  # sort ascending
	for idx in sort_indices[::-1]:
		upos = UPOS_TAGSET[idx]
		upos_stats = upos_to_correctness[upos]
		num_total = len(upos_stats['correct']) + len(upos_stats['incorrect'])

		print(f"{upos} correctness = {correctness[idx]:.4f} (N = {num_total})")
		print("Most frequently correct words:")
		print(Counter(upos_stats["correct"]).most_common())
		print("Most frequently incorrect words:")
		print(Counter(upos_stats["incorrect"]).most_common())
		print("")

	import matplotlib.pyplot as plt
	indexer = np.arange(len(UPOS_TAGSET))
	plt.figure(figsize=(7, 7))
	plt.title("Token-level accuracy by UPOS")
	plt.barh(indexer, [correctness[_i] for _i in sort_indices])
	plt.yticks(indexer, [UPOS_TAGSET[_i] for _i in sort_indices])
	plt.xlim([0, 1.0 + 0.01])
	plt.xlabel("Accuracy")

	for _i in indexer:
		idx_tag = sort_indices[_i]
		upos = UPOS_TAGSET[idx_tag]
		num_total = len(upos_to_correctness[upos]["correct"]) + len(upos_to_correctness[upos]["incorrect"])
		curr_acc = correctness[idx_tag]

		plt.text(curr_acc, _i - 0.25, f"{curr_acc:.3f} (N={num_total})")

	plt.tight_layout()
	plt.savefig("accuracy_by_upos.png")
















