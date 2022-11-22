from collections import Counter

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from data import load_df

if __name__ == "__main__":
	model_results_paths = [
		"RESULTS/WORD/komet-csebert-2e-5-binary3-history0-optthresh/komet-fold0-csebert-2e-5-binary3-history0-optthresh/predict-test/test_results.tsv",
		"RESULTS/WORD/komet-csebert-2e-5-binary3-history0-optthresh/komet-fold1-csebert-2e-5-binary3-history0-optthresh/predict-test/test_results.tsv",
		"RESULTS/WORD/komet-csebert-2e-5-binary3-history0-optthresh/komet-fold2-csebert-2e-5-binary3-history0-optthresh/predict-test/test_results.tsv",
		"RESULTS/WORD/komet-csebert-2e-5-binary3-history0-optthresh/komet-fold3-csebert-2e-5-binary3-history0-optthresh/predict-test/test_results.tsv",
		"RESULTS/WORD/komet-csebert-2e-5-binary3-history0-optthresh/komet-fold4-csebert-2e-5-binary3-history0-optthresh/predict-test/test_results.tsv",
	]

	df = pd.concat([load_df(curr_path) for curr_path in model_results_paths], axis=0).reset_index(drop=True)
	df["num_met_words"] = df["met_type"].apply(lambda _mets: sum([len(met_info["word_indices"]) for met_info in _mets]))
	df["prop_met_words"] = df["num_met_words"] / df["sentence_words"].apply(lambda _words: len(_words))

	# These are defined by feel, making sure that each bin has enough samples to infer something from it
	def _discretize(_p):
		if _p == 0.0:
			return "0.0"
		elif 0.0 < _p <= 0.1:
			return "(0.0, 0.1]"
		elif 0.1 < _p <= 0.2:
			return "(0.1, 0.2]"
		else:
			return "(0.2, 1]"
	df["prop_met_words"] = df["prop_met_words"].apply(_discretize)

	def token_correctness(_preds, _true):
		return Counter([_p == _t for _p, _t in zip(_preds, _true) if _t not in {"O", 0, "not_metaphor"}])  # RECALL!

	def sentence_correctness(_preds, _true):
		return Counter([_preds == _true])  # ACCURACY

	plot_desc = None
	groups = ["0.0", "(0.0, 0.1]", "(0.1, 0.2]", "(0.2, 1]"]
	correctness_stats = []
	total_counted = []
	for curr_prop in groups:
		curr_group = df.loc[df["prop_met_words"] == curr_prop]

		print(f"{curr_prop}: {curr_group.shape[0]} examples")
		group_correctness = Counter()
		for idx_ex in range(curr_group.shape[0]):
			curr_ex = curr_group.iloc[idx_ex]
			curr_preds, curr_true = curr_ex["preds_transformed"], curr_ex["true_transformed"]

			if isinstance(curr_preds, list) and isinstance(curr_true, list):
				plot_desc = "Token-level metaphor recall"
				group_correctness += token_correctness(curr_preds, curr_true)
			elif isinstance(curr_preds, str) and isinstance(curr_true, str):
				plot_desc = "Sentence-level accuracy"
				group_correctness += sentence_correctness(curr_preds, curr_true)
			else:
				raise ValueError(f"Encountered unexpected combination of types: {type(curr_preds)} and {type(curr_true)}")

		denom = group_correctness[True] + group_correctness[False]
		correctness_stats.append(group_correctness[True] / max(1, denom))
		total_counted.append(denom)

	indexer = np.arange(len(groups))
	plt.figure(figsize=(7, 5))
	plt.title(f"{plot_desc} by number of met. words in a sentence")
	plt.barh(indexer, correctness_stats)
	plt.yticks(indexer, groups)
	plt.ylabel("Proportion of met. words in a sentence", rotation="vertical")
	plt.xlabel("Accuracy")

	for _i in indexer:
		plt.text(correctness_stats[_i], _i - 0.05, f"{correctness_stats[_i]:.3f} (N={total_counted[_i]})")

	plt.tight_layout()
	plt.show()

