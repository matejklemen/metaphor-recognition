import argparse
import json
import logging
import os.path
import sys

import numpy as np
import torch
import wandb
from transformers import AutoTokenizer, AutoModelForTokenClassification

from base import MetaphorController
from data import load_df, TAG2ID, TransformersTokenDataset

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="debug")

parser.add_argument("--train_path", type=str, default="data/train_data.tsv")
parser.add_argument("--dev_path", type=str, default="data/dev_data.tsv")
parser.add_argument("--test_path", type=str, default="data/test_data.tsv")

# <option>_N indicates N labels being taken into account, others are treated as "other"
# Priority: MRWi, MRWd, WIDLI, MFlag
# Example: independent_3 will encode MRWi, MRWd, WIDLI separately, and treat MFlag same as no metaphor
parser.add_argument("--label_scheme", type=str, default="binary_2",
					choices=["binary_1", "binary_2", "binary_3", "binary_4",
							 "independent_1", "independent_2", "independent_3", "independent_4"])
parser.add_argument("--iob2", action="store_true", help="Encode labels with IOB2 label scheme")

parser.add_argument("--pretrained_name_or_path", type=str, default="EMBEDDIA/sloberta")
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--max_length", type=int, default=32)
parser.add_argument("--stride", type=int, default=None,
					help="When examples are longer than `max_length`, examples get broken up into multiple examples "
						 "with first `stride` subwords overlapping")
parser.add_argument("--history_prev_sents", type=int, default=0,
					help="Number of previous sentences to take as additional context")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--validate_steps", type=int, default=3000)

parser.add_argument("--random_seed", type=int, default=None)
parser.add_argument("--wandb_project_name", type=str, default="metaphor-detection-komet")
parser.add_argument("--use_cpu", action="store_true")

if __name__ == "__main__":
	args = parser.parse_args()

	if not os.path.exists(args.model_dir):
		os.makedirs(args.model_dir)

	# Set up logging to file and stdout
	logger = logging.getLogger()
	logger.setLevel(logging.INFO)
	for curr_handler in [logging.StreamHandler(sys.stdout),
						 logging.FileHandler(os.path.join(args.model_dir, "training.log"))]:
		curr_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s"))
		logger.addHandler(curr_handler)

	if not args.use_cpu and not torch.cuda.is_available():
		logging.info(f"No CUDA device found, overriding `--use_cpu` flag")
		args.use_cpu = True

	for k, v in vars(args).items():
		v_str = str(v)
		v_str = f"...{v_str[-(50 - 3):]}" if len(v_str) > 50 else v_str
		logging.info(f"|{k:30s}|{v_str:50s}|")

	with open(os.path.join(args.model_dir, "training_args.json"), "w", encoding="utf-8") as f_conf:
		json.dump(vars(args), fp=f_conf, indent=4)

	if args.random_seed is not None:
		torch.manual_seed(args.random_seed)
		np.random.seed(args.random_seed)

	DEVICE = torch.device("cpu") if args.use_cpu else torch.device("cuda")
	STRIDE = args.max_length // 2 if args.stride is None else args.stride
	args.stride = STRIDE
	# Convert from e.g., "binary_2" -> "binary", "2"
	TYPE_LABEL_SCHEME, NUM_LABELS = args.label_scheme.split("_")
	PRIMARY_LABEL_SCHEME = TYPE_LABEL_SCHEME
	SECONDARY_LABEL_SCHEME = TYPE_LABEL_SCHEME  # If IOB2 is used, holds the name of the non-IOB2 equivalent scheme
	NUM_LABELS = int(NUM_LABELS)
	wandb.init(project=args.wandb_project_name, config=vars(args))

	# iob2 transforms each positive label into two labels, e.g., metaphor -> {B-metaphor, I-metaphor}
	FALLBACK_LABEL = "O"
	if PRIMARY_LABEL_SCHEME == "binary":
		FALLBACK_LABEL = "not_metaphor"
	elif PRIMARY_LABEL_SCHEME == "independent":
		FALLBACK_LABEL = "O"

	if args.iob2:
		PRIMARY_LABEL_SCHEME, SECONDARY_LABEL_SCHEME = f"{PRIMARY_LABEL_SCHEME}_iob2", PRIMARY_LABEL_SCHEME

	train_df = load_df(args.train_path)
	dev_df = load_df(args.dev_path)
	test_df = load_df(args.test_path)

	tokenizer = AutoTokenizer.from_pretrained(args.pretrained_name_or_path)
	model = AutoModelForTokenClassification.from_pretrained(args.pretrained_name_or_path,
															num_labels=len(TAG2ID[PRIMARY_LABEL_SCHEME])).to(DEVICE)
	controller = MetaphorController(
		model_dir=args.model_dir,
		label_scheme=PRIMARY_LABEL_SCHEME, tokenizer_or_tokenizer_name=tokenizer, model_or_model_name=model,
		learning_rate=args.learning_rate, batch_size=args.batch_size,
		validate_every_n_examples=args.validate_steps, optimized_metric="f1_macro",
		device=("cpu" if args.use_cpu else "cuda")
	)

	# ------------------------------------
	train_dataset = TransformersTokenDataset.from_dataframe(
		train_df, label_scheme=args.label_scheme, primary_label_scheme=PRIMARY_LABEL_SCHEME,
		max_length=args.max_length, stride=STRIDE, history_prev_sents=args.history_prev_sents,
		fallback_label=FALLBACK_LABEL, iob2=args.iob2, tokenizer_or_tokenizer_name=tokenizer
	)

	# ------------------------------------

	dev_dataset = TransformersTokenDataset.from_dataframe(
		dev_df, label_scheme=args.label_scheme, primary_label_scheme=PRIMARY_LABEL_SCHEME,
		max_length=args.max_length, stride=STRIDE, history_prev_sents=args.history_prev_sents,
		fallback_label=FALLBACK_LABEL, iob2=args.iob2, tokenizer_or_tokenizer_name=tokenizer
	)

	# ------------------------------------
	logging.info(f"Loaded {len(train_dataset)} train examples, {len(dev_dataset)} dev examples")
	controller.run_training(train_dataset, dev_dataset, num_epochs=args.num_epochs)
	wandb.finish()
