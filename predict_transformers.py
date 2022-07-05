import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch

from base import MetaphorController
from data import load_df, TransformersTokenDataset, LOSS_IGNORE_INDEX, ID2TAG, TAG2ID
from utils import visualize_token_predictions

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_dir", type=str, default="debug/debug-experiment")

parser.add_argument("--model_dir", type=str, default="debug")
parser.add_argument("--test_path", type=str, default="data/komet/test_data.tsv")

parser.add_argument("--mcd_iters", type=int, default=0)

# These are optional arguments, which get inferred from the model's training arguments in case they are not provided
parser.add_argument("--max_length", type=int, default=None)
parser.add_argument("--stride", type=int, default=None)
parser.add_argument("--batch_size", type=int, default=None)

parser.add_argument("--random_seed", type=int, default=None)
parser.add_argument("--use_cpu", action="store_true")

if __name__ == "__main__":
	args = parser.parse_args()

	if not os.path.exists(args.experiment_dir):
		os.makedirs(args.experiment_dir)

	# Set up logging to file and stdout
	logger = logging.getLogger()
	logger.setLevel(logging.INFO)
	for curr_handler in [logging.StreamHandler(sys.stdout),
						 logging.FileHandler(os.path.join(args.experiment_dir, "evaluation.log"))]:
		curr_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s"))
		logger.addHandler(curr_handler)

	with open(os.path.join(args.model_dir, "training_args.json"), "r", encoding="utf-8") as f_args:
		pred_args = json.load(f_args)

	if args.max_length is not None:
		new_stride = args.max_length // 2 if args.stride is None else args.stride
		pred_args["max_length"] = args.max_length
		pred_args["stride"] = new_stride

	if args.batch_size is not None:
		pred_args["batch_size"] = args.batch_size

	if args.random_seed is not None:
		pred_args["random_seed"] = args.random_seed

	if args.use_cpu:
		pred_args["use_cpu"] = args.use_cpu

	if not args.use_cpu and not torch.cuda.is_available():
		logging.info(f"No CUDA device found, overriding `--use_cpu` flag")
		pred_args["use_cpu"] = True

	for k, v in pred_args.items():
		v_str = str(v)
		v_str = f"...{v_str[-(50 - 3):]}" if len(v_str) > 50 else v_str
		logging.info(f"|{k:30s}|{v_str:50s}|")

	if args.random_seed is not None:
		torch.manual_seed(args.random_seed)
		np.random.seed(args.random_seed)

	test_fname = args.test_path.split(os.path.sep)[-1]
	# Save test data in the experiment (for reference)
	test_df = load_df(args.test_path)
	test_df.to_csv(os.path.join(args.experiment_dir, f"raw_{test_fname}"), sep="\t", index=False)

	controller = MetaphorController.load(args.model_dir,
										 batch_size=pred_args["batch_size"],
										 device=("cpu" if pred_args["use_cpu"] else "cuda"))
	test_dataset = TransformersTokenDataset.from_dataframe(
		test_df, label_scheme=controller.label_scheme,
		max_length=pred_args["max_length"], stride=pred_args["stride"],
		history_prev_sents=pred_args["history_prev_sents"], tokenizer_or_tokenizer_name=controller.tokenizer
	)

	processed_test_df = pd.DataFrame({"sentence_words": test_dataset.sample_words})

	test_res = controller.run_prediction(test_dataset, mcd_iters=args.mcd_iters)
	test_preds = test_dataset.align_word_predictions(test_res["preds"].cpu(), pad=False)

	test_true = None
	if test_dataset.has_labels():
		test_true = test_dataset.labels

		if controller.scheme_info["iob2"]:
			independent_labels = []
			for idx_ex in range(test_true.shape[0]):
				curr_labels = test_true[idx_ex]
				curr_processed = []
				for _lbl in curr_labels.tolist():
					if _lbl == LOSS_IGNORE_INDEX:
						curr_processed.append(_lbl)
					else:
						str_tag = ID2TAG[controller.prim_label_scheme][_lbl]
						if str_tag == controller.fallback_str:
							curr_processed.append(TAG2ID[controller.sec_label_scheme][str_tag])
						else:
							# without "B-" or "I-"
							curr_processed.append(TAG2ID[controller.sec_label_scheme][str_tag[2:]])

				independent_labels.append(curr_processed)

			independent_labels = torch.tensor(independent_labels)
			assert independent_labels.shape == test_true.shape
			test_true = independent_labels

		test_true_meta = test_dataset.align_word_predictions(test_true, pad=False)
		test_true_meta = [list(map(lambda _curr_lbl: ID2TAG[controller.sec_label_scheme].get(_curr_lbl, _curr_lbl), _curr_true))
						  for _curr_true in test_true_meta]
		processed_test_df["true_met_type"] = test_true_meta

		test_probas = test_res["pred_probas"].cpu()
		if controller.iob2:
			test_probas = None
		test_metrics = controller.compute_metrics(test_dataset,
												  pred_labels=test_res["preds"].cpu(),
												  true_labels=test_true,
												  pred_probas=test_probas)

		test_metrics_verbose = []
		for metric_name, metric_val in sorted(test_metrics.items(), key=lambda tup: tup[0]):
			test_metrics_verbose.append(f"{metric_name} = {metric_val:.4f}")
		test_metrics_verbose = "[Test metrics] {}".format(", ".join(test_metrics_verbose))
		logging.info(test_metrics_verbose)

	test_preds = [list(map(lambda _curr_lbl: ID2TAG[controller.sec_label_scheme].get(_curr_lbl, _curr_lbl), _curr_preds))
				  for _curr_preds in test_preds]
	processed_test_df["pred_met_type"] = test_preds

	with open(os.path.join(args.experiment_dir, "pred_visualization.html"), "w", encoding="utf-8") as f:
		visualization_html = visualize_token_predictions(test_dataset.sample_words, test_preds, test_true)
		print(visualization_html, file=f)

	with open(os.path.join(args.experiment_dir, "prediction_args.json"), "w", encoding="utf-8") as f:
		json.dump(pred_args, fp=f, indent=4)

	processed_test_df.to_csv(os.path.join(args.experiment_dir, f"processed_{test_fname}"), sep="\t", index=False)
