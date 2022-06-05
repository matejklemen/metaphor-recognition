import argparse
import json
import logging
import os.path
import sys
from collections import Counter

import numpy as np
import torch
from transformers import AutoTokenizer

import wandb
from custom_modules import AutoModelForTokenMultiClassification
from data import load_df, extract_scheme_info, TransformersTokenDatasetWithFrames, \
	FALLBACK_LABEL_INDEX
from multitask import MetaphorMultiTaskController

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="debug")

parser.add_argument("--train_path", type=str, default="data/komet/train_data.tsv")
parser.add_argument("--dev_path", type=str, default="data/komet/dev_data.tsv")

# <option>_N indicates N labels being taken into account, others are treated as "other"
# Priority: MRWi, MRWd, WIDLI, MFlag
# Example: independent_3 will encode MRWi, MRWd, WIDLI separately, and treat MFlag same as no metaphor
parser.add_argument("--label_scheme", type=str, default="binary_2",
					choices=["binary_1", "binary_2", "binary_3", "binary_4",
							 "independent_1", "independent_2", "independent_3", "independent_4"])
parser.add_argument("--iob2", action="store_true", help="Encode labels with IOB2 label scheme")
parser.add_argument("--min_frame_freq", type=int, default=100)

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

	STRIDE = args.max_length // 2 if args.stride is None else args.stride
	args.stride = STRIDE
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
	if args.iob2:
		args.label_scheme = f"{args.label_scheme}_iob2"
	scheme_info = extract_scheme_info(args.label_scheme)
	num_train_mtypes = 1 + scheme_info["primary"]["num_pos_labels"]  # includes fallback (negative) label

	train_df = load_df(args.train_path)
	train_df["met_frame"] = [
		[_frame if _frame == "O" else _frame.split("/")[0] for _frame in curr_frames]
		for curr_frames in train_df["met_frame"].tolist()
	]
	frame_counter = Counter()
	for curr_frames in train_df["met_frame"].tolist():
		frame_counter += Counter(curr_frames)

	frame_encoding = {"O": FALLBACK_LABEL_INDEX}
	MIN_FRAME_FREQ = args.min_frame_freq
	for frame_type, count in frame_counter.items():
		if count >= MIN_FRAME_FREQ and frame_type not in frame_encoding:
			frame_encoding[frame_type] = len(frame_encoding)
	# TODO: this wont work with BIO (needs a fix)
	num_train_mframes = len(frame_encoding)
	if args.iob2:
		raise NotImplementedError("Logic for using IOB2 with metaphor frames is not implemented yet")
	logging.info(f"Using {len(frame_encoding)} frame types:\n{frame_encoding}")

	dev_df = load_df(args.dev_path)
	dev_df["met_frame"] = [
		[_frame if _frame == "O" else _frame.split("/")[0] for _frame in curr_frames]
		for curr_frames in dev_df["met_frame"].tolist()
	]

	tokenizer = AutoTokenizer.from_pretrained(args.pretrained_name_or_path)
	model = AutoModelForTokenMultiClassification(args.pretrained_name_or_path,
												 num_types=num_train_mtypes,
												 num_frames=num_train_mframes).to(DEVICE)

	train_dataset = TransformersTokenDatasetWithFrames.from_dataframe(
		train_df, label_scheme=args.label_scheme, max_length=args.max_length, stride=STRIDE,
		history_prev_sents=args.history_prev_sents, tokenizer_or_tokenizer_name=tokenizer,
		frame_encoding=frame_encoding
	)

	dev_dataset = TransformersTokenDatasetWithFrames.from_dataframe(
		dev_df, label_scheme=args.label_scheme, max_length=args.max_length, stride=STRIDE,
		history_prev_sents=args.history_prev_sents, tokenizer_or_tokenizer_name=tokenizer,
		frame_encoding=frame_encoding
	)

	controller = MetaphorMultiTaskController(
		model_dir=args.model_dir, label_scheme=args.label_scheme,
		tokenizer_or_tokenizer_name=tokenizer, model_or_model_name=model,
		learning_rate=args.learning_rate, batch_size=args.batch_size,
		validate_every_n_examples=args.validate_steps, optimized_metric="f1_macro",
		device=("cpu" if args.use_cpu else "cuda"), frame_encoding=frame_encoding
	)

	# ------------------------------------
	logging.info(f"Loaded {len(train_dataset)} train examples, {len(dev_dataset)} dev examples")
	controller.run_training(train_dataset, dev_dataset, num_epochs=args.num_epochs)
	wandb.finish()
