import argparse
import os
from collections import Counter

import numpy as np
from matplotlib import pyplot as plt

from data import load_df

if __name__ == "__main__":
	model_results_dir = [
		"RESULTS/SENTENCE/komet-sent-csebert-2e-5-binary3-history0-optthresh"
	]
	short_names = [
		"csebert"
	]
	systems_indexer = np.arange(len(model_results_dir))
	cmap = plt.cm.get_cmap("tab10")
	assert len(model_results_dir) <= 10, "If you need more than 10 colors, remove assert and change cmap to 'tab20'"
	system_colors = [cmap(i) for i in range(len(model_results_dir))]

	file_name = "test_results.tsv"

	num_mets_to_stats = {}  # 0 -> {system1: ..., system2: ...}, 1 -> {...}
	num_mets_to_total = {}

	for model_dir, model_name in zip(model_results_dir, short_names):
		print(f"'{model_name}'")
		results_dir = [os.path.join(model_dir, dirname) for dirname in os.listdir(model_dir)
					   if os.path.isdir(os.path.join(model_dir, dirname))]
		results_dir = sorted(results_dir)

		num_metaphors_correctness = {}
		for curr_dir in results_dir:
			file_path = os.path.join(curr_dir, file_name)
			df = load_df(file_path)

			df["num_metaphors"] = df["met_type"].apply(lambda _mets: len(_mets))
			for num_mets, curr_group in df.groupby("num_metaphors"):
				existing_count = num_metaphors_correctness.get(num_mets, Counter())
				existing_count += Counter((curr_group["preds_transformed"] == curr_group["true_transformed"]).tolist())
				num_metaphors_correctness[num_mets] = existing_count

		for num, correctness in num_metaphors_correctness.items():
			existing_stats = num_mets_to_stats.get(num, {})
			existing_stats[model_name] = correctness[True] / (correctness[True] + correctness[False])
			num_mets_to_stats[num] = existing_stats

			num_mets_to_total[num] = (correctness[True] + correctness[False])

	NUM_UNIQ_PLOTS = len(num_mets_to_stats.keys())
	SUBPLOTS_PER_ROW = 4
	num_vis_rows = (NUM_UNIQ_PLOTS + SUBPLOTS_PER_ROW - 1) // SUBPLOTS_PER_ROW

	plt.figure(figsize=(20, 35))
	plt.suptitle("Sentence-level accuracy by number of annotated metaphors in sentence")
	for _i, (num_met, accuracies) in enumerate(
			sorted(num_mets_to_stats.items(), key=lambda tup: tup[0]), start=1
	):
		curr_total = num_mets_to_total[num_met]
		sys_names = list(map(lambda tup: tup[0], accuracies.items()))
		sys_accs = list(map(lambda tup: tup[1], accuracies.items()))

		sort_indices = np.argsort(-np.array(sys_accs))
		plt.subplot(num_vis_rows, SUBPLOTS_PER_ROW, _i)
		plt.title(f"num_met={num_met} (N={curr_total})", pad=10, fontsize=12)
		plt.barh(systems_indexer, [sys_accs[_i] for _i in sort_indices], height=0.1,
				 color=[system_colors[_i] for _i in sort_indices])
		plt.xlim([0.0, 1.0 + 0.05])
		plt.xticks(np.arange(0.0, 1.0 + 0.05, 0.1))
		plt.yticks(systems_indexer, [sys_names[_i] for _i in sort_indices])

		for i in range(len(systems_indexer)):
			curr_acc = sys_accs[sort_indices[i]]
			plt.text(curr_acc, i, f"{curr_acc:.3f}")

		print(f"num_mets = {num_met}")
		print(sys_names)
		print(sys_accs)

	plt.tight_layout()
	plt.subplots_adjust(top=0.9, hspace=0.5)
	plt.show()
