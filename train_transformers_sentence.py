import argparse
import json
import logging
import os.path
import sys

import numpy as np
import torch
import wandb
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from base import MetaphorSentenceController
from data import load_df, extract_scheme_info, TransformersSentenceDataset

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="debug")

parser.add_argument("--train_path", type=str, default="data/komet/train_data.tsv")
parser.add_argument("--dev_path", type=str, default="data/komet/dev_data.tsv")

# <option>_N indicates N labels being taken into account, others are treated as "other"
# Priority: MRWi, MRWd, WIDLI, MFlag
# Example: independent_3 will encode MRWi, MRWd, WIDLI separately, and treat MFlag same as no metaphor
parser.add_argument("--label_scheme", type=str, default="binary_2")

parser.add_argument("--pretrained_name_or_path", type=str, default="EMBEDDIA/sloberta")
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--max_length", type=int, default=32)
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

	DEVICE = torch.device("cpu") if args.use_cpu else torch.device("cuda")

	for k, v in vars(args).items():
		v_str = str(v)
		v_str = f"...{v_str[-(50 - 3):]}" if len(v_str) > 50 else v_str
		logging.info(f"|{k:30s}|{v_str:50s}|")

	with open(os.path.join(args.model_dir, "training_args.json"), "w", encoding="utf-8") as f_conf:
		json.dump(vars(args), fp=f_conf, indent=4)

	if args.random_seed is not None:
		torch.manual_seed(args.random_seed)
		np.random.seed(args.random_seed)

	wandb.init(project=args.wandb_project_name, config=vars(args))
	scheme_info = extract_scheme_info(args.label_scheme)
	num_train_labels = 1 + scheme_info["primary"]["num_pos_labels"]  # includes fallback (negative) label

	train_df = load_df(args.train_path)
	dev_df = load_df(args.dev_path)

	if "roberta" in args.pretrained_name_or_path:
		# Hack, RoBERTa models need a specific tokenizer argument
		tokenizer = AutoTokenizer.from_pretrained(args.pretrained_name_or_path, add_prefix_space=True)
	else:
		tokenizer = AutoTokenizer.from_pretrained(args.pretrained_name_or_path)
	model = AutoModelForSequenceClassification.from_pretrained(args.pretrained_name_or_path,
															   num_labels=num_train_labels).to(DEVICE)
	controller = MetaphorSentenceController(
		model_dir=args.model_dir, label_scheme=args.label_scheme,
		tokenizer_or_tokenizer_name=tokenizer, model_or_model_name=model,
		learning_rate=args.learning_rate, batch_size=args.batch_size,
		validate_every_n_examples=args.validate_steps, optimized_metric="f1_macro",
		device=("cpu" if args.use_cpu else "cuda")
	)

	train_dataset = TransformersSentenceDataset.from_dataframe(
		train_df, label_scheme=args.label_scheme, max_length=args.max_length,
		history_prev_sents=args.history_prev_sents, tokenizer_or_tokenizer_name=tokenizer
	)

	dev_dataset = TransformersSentenceDataset.from_dataframe(
		dev_df, label_scheme=args.label_scheme, max_length=args.max_length,
		history_prev_sents=args.history_prev_sents, tokenizer_or_tokenizer_name=tokenizer
	)

	# ------------------------------------
	logging.info(f"Loaded {len(train_dataset)} train examples, {len(dev_dataset)} dev examples")
	controller.run_training(train_dataset, dev_dataset, num_epochs=args.num_epochs)
	wandb.finish()
