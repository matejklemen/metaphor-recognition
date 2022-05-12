import argparse
import os.path
import sys

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification

from data import load_df, create_examples, TAG2ID, LOSS_IGNORE_INDEX, TransformersSeqDataset, ID2TAG
from utils import token_precision, token_recall, token_f1

import logging

# TODO: model saving based on validation metric
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="debug")

parser.add_argument("--train_path", type=str, default="data/train_data.tsv")
parser.add_argument("--dev_path", type=str, default="data/dev_data.tsv")
parser.add_argument("--test_path", type=str, default="data/test_data.tsv")

parser.add_argument("--label_scheme", type=str, default="simple",
					choices=["simple"])

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
	DEV_BATCH_SIZE = 2 * args.batch_size  # no grad computation
	SUBSET_SIZE = args.validate_steps
	STRIDE = args.max_length // 2 if args.stride is None else args.stride

	# TODO: adapt pos_label based on scheme
	POS_LABEL = [1]

	train_df = load_df(args.train_path)
	dev_df = load_df(args.dev_path)
	test_df = load_df(args.test_path)

	tokenizer = AutoTokenizer.from_pretrained(args.pretrained_name_or_path)
	model = AutoModelForTokenClassification.from_pretrained(args.pretrained_name_or_path,
															num_labels=len(TAG2ID[args.label_scheme])).to(DEVICE)
	optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

	# ------------------------------------

	orig_train_lbls = train_df["met_type"].tolist()
	if args.label_scheme == "simple":
		train_df["met_type"] = train_df["met_type"].apply(lambda _fine_labels:
														  list(map(lambda _lbl: "metaphor" if _lbl in ("MRWd", "MRWi") else "not_metaphor", _fine_labels)))

	train_in, train_out = create_examples(train_df,
										  encoding_scheme=TAG2ID[args.label_scheme],
										  history_prev_sents=args.history_prev_sents,
										  fallback_label="not_metaphor")

	num_train = len(train_in)
	enc_train_in = tokenizer(
		train_in, is_split_into_words=True,
		max_length=args.max_length, padding="max_length", truncation=True,
		return_overflowing_tokens=True, stride=STRIDE,
		return_tensors="pt"
	)

	is_start_encountered = np.zeros(num_train, dtype=bool)
	enc_train_out = []
	for idx_ex, (curr_input_ids, idx_orig_ex) in enumerate(zip(enc_train_in["input_ids"],
															   enc_train_in["overflow_to_sample_mapping"])):
		curr_word_ids = enc_train_in.word_ids(idx_ex)

		# where does sequence actually start, i.e. after <bos>
		nonspecial_start = 0
		while curr_word_ids[nonspecial_start] is not None:
			nonspecial_start += 1

		# when an example is broken up, all but the first sub-example have first `stride` tokens overlapping with prev.
		ignore_n_overlapping = 0
		if is_start_encountered[idx_orig_ex]:
			ignore_n_overlapping = STRIDE
		else:
			is_start_encountered[idx_orig_ex] = True

		fixed_out = []
		fixed_out += [LOSS_IGNORE_INDEX] * (nonspecial_start + ignore_n_overlapping)

		for idx_subw, w_id in enumerate(curr_word_ids[(nonspecial_start + ignore_n_overlapping):],
										start=(nonspecial_start + ignore_n_overlapping)):
			if curr_word_ids[idx_subw] is None:
				fixed_out.append(LOSS_IGNORE_INDEX)
			else:
				fixed_out.append(train_out[idx_orig_ex][w_id])

		enc_train_out.append(fixed_out)

	enc_train_in["labels"] = torch.tensor(enc_train_out)
	train_dataset = TransformersSeqDataset(**enc_train_in)

	# ------------------------------------

	orig_dev_lbls = dev_df["met_type"].tolist()
	if args.label_scheme == "simple":
		dev_df["met_type"] = dev_df["met_type"].apply(lambda _fine_labels:
													  list(map(lambda _lbl: "metaphor" if _lbl in ("MRWd", "MRWi") else "not_metaphor", _fine_labels)))

	dev_in, dev_out = create_examples(dev_df,
									  encoding_scheme=TAG2ID[args.label_scheme],
									  history_prev_sents=args.history_prev_sents,
									  fallback_label="not_metaphor")

	num_dev = len(dev_in)
	enc_dev_in = tokenizer(
		dev_in, is_split_into_words=True,
		max_length=args.max_length, padding="max_length", truncation=True,
		return_overflowing_tokens=True, stride=STRIDE,
		return_tensors="pt"
	)

	is_start_encountered = np.zeros(num_dev, dtype=bool)
	enc_dev_out = []
	for idx_ex, (curr_input_ids, idx_orig_ex) in enumerate(zip(enc_dev_in["input_ids"],
															   enc_dev_in["overflow_to_sample_mapping"])):
		curr_word_ids = enc_dev_in.word_ids(idx_ex)

		# where does sequence actually start, i.e. after <bos>
		nonspecial_start = 0
		while curr_word_ids[nonspecial_start] is not None:
			nonspecial_start += 1

		# when an example is broken up, all but the first sub-example have first `stride` tokens overlapping with prev.
		ignore_n_overlapping = 0
		if is_start_encountered[idx_orig_ex]:
			ignore_n_overlapping = STRIDE
		else:
			is_start_encountered[idx_orig_ex] = True

		fixed_out = []
		fixed_out += [LOSS_IGNORE_INDEX] * (nonspecial_start + ignore_n_overlapping)

		for idx_subw, w_id in enumerate(curr_word_ids[(nonspecial_start + ignore_n_overlapping):],
										start=(nonspecial_start + ignore_n_overlapping)):
			if curr_word_ids[idx_subw] is None:
				fixed_out.append(LOSS_IGNORE_INDEX)
			else:
				fixed_out.append(dev_out[idx_orig_ex][w_id])

		enc_dev_out.append(fixed_out)

	enc_dev_in["labels"] = torch.tensor(enc_dev_out)
	dev_dataset = TransformersSeqDataset(**enc_dev_in)
	num_dev_batches = (len(dev_dataset) + DEV_BATCH_SIZE - 1) // DEV_BATCH_SIZE

	# ------------------------------------
	logging.info(f"Loaded {len(train_dataset)} train examples, {len(dev_dataset)} dev examples")

	num_train_subsets = (len(train_dataset) + SUBSET_SIZE - 1) // SUBSET_SIZE
	for idx_epoch in range(args.num_epochs):
		logging.info(f"Epoch #{1 + idx_epoch}/{args.num_epochs}")
		train_loss, nb = 0.0, 0

		rand_indices = torch.randperm(num_train)
		for idx_subset in range(num_train_subsets):
			s_sub, e_sub = idx_subset * SUBSET_SIZE, (idx_subset + 1) * SUBSET_SIZE
			curr_sub = Subset(train_dataset, rand_indices[s_sub: e_sub])

			model.train()
			for curr_batch_cpu in tqdm(DataLoader(curr_sub, batch_size=args.batch_size)):
				curr_batch = {_k: _v.to(DEVICE) for _k, _v in curr_batch_cpu.items()}

				res = model(**curr_batch)
				loss = res["loss"]

				train_loss += float(loss)
				nb += 1

				loss.backward()
				optimizer.step()
				optimizer.zero_grad()

			logging.info(f"Training loss: {train_loss / nb: .4f}")

			dev_loss = 0.0
			dev_preds = []
			with torch.no_grad():
				model.eval()
				for curr_batch_cpu in tqdm(DataLoader(dev_dataset, batch_size=DEV_BATCH_SIZE)):
					curr_batch = {_k: _v.to(DEVICE) for _k, _v in curr_batch_cpu.items()}

					res = model(**curr_batch)
					preds = torch.argmax(torch.softmax(res["logits"], dim=-1), dim=-1).cpu()

					dev_loss += float(res["loss"])
					dev_preds.append(preds)

			dev_preds = torch.cat(dev_preds)

			logging.info(f"Dev loss: {dev_loss / num_dev_batches: .4f}")
			for curr_label in POS_LABEL:
				curr_label_str = ID2TAG[args.label_scheme][curr_label]
				dev_p = token_precision(dev_dataset.labels, dev_preds, pos_label=1)
				dev_r = token_recall(dev_dataset.labels, dev_preds, pos_label=1)
				dev_f1 = token_f1(dev_dataset.labels, dev_preds, pos_label=1)
				logging.info(f"[{curr_label_str}] dev P={dev_p:.3f}, R={dev_r:.3f}, F1={dev_f1:.3f}")

