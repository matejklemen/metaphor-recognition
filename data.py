import ast
import itertools
from typing import Dict, Optional, List, Iterable, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from utils import preprocess_iob2

POS_MET_TYPES = ["MRWi", "MRWd", "WIDLI", "MFlag"]
TAG2ID = {
	"binary": {_tag: _i for _i, _tag in enumerate(["not_metaphor", "metaphor"])},
	"binary_iob2": {_tag: _i for _i, _tag in enumerate(["not_metaphor"] + list(map(lambda tup: "".join(tup),
																				   itertools.product(["B-", "I-"], ["metaphor"]))))},
	"independent": {_tag: _i for _i, _tag in enumerate(["O"] + POS_MET_TYPES)},
	"independent_iob2": {_tag: _i for _i, _tag in enumerate(["O"] + list(map(lambda tup: "".join(tup),
																			 itertools.product(["B-", "I-"], POS_MET_TYPES))))}
}

ID2TAG = {curr_scheme: {_i: _tag for _tag, _i in TAG2ID[curr_scheme].items()} for curr_scheme in TAG2ID}
# This is used to mark labels that should not be taken into account in loss calculation
LOSS_IGNORE_INDEX = -100
# IMPORTANT: Assuming fallback label is encoded with 0 in every label scheme
FALLBACK_LABEL_INDEX = 0


class TransformersTokenDataset(Dataset):
	def __init__(self, **kwargs):
		self.metadata_attrs = ["subsample_to_sample", "subword_to_word"]
		self.valid_attrs = []
		for attr, values in kwargs.items():
			if attr not in self.metadata_attrs:
				self.valid_attrs.append(attr)
			setattr(self, attr, values)

		assert len(self.valid_attrs) > 0
		# Need for alignment of "subsamples" to samples
		assert all(hasattr(self, _attr_name) for _attr_name in ["subsample_to_sample", "subword_to_word"])
		self.num_unique_samples = len(set(kwargs["subsample_to_sample"]))

		self.num_words_in_sample = [0 for _ in range(self.num_unique_samples)]
		for idx_sample, word_indices in zip(getattr(self, "subsample_to_sample"),
											getattr(self, "subword_to_word")):
			curr_max = self.num_words_in_sample[idx_sample]
			self.num_words_in_sample[idx_sample] = max(curr_max,
													   1 + max(filter(lambda w_id: w_id is not None, word_indices), default=-1))

	def __getitem__(self, item):
		return {k: getattr(self, k)[item] for k in self.valid_attrs}

	def __len__(self):
		return len(getattr(self, self.valid_attrs[0]))

	def has_labels(self):
		return hasattr(self, "labels")

	@staticmethod
	def from_dataframe(df, label_scheme, max_length, stride, history_prev_sents=0,
					   tokenizer_or_tokenizer_name: Union[str, AutoTokenizer] = "EMBEDDIA/sloberta"):
		# TODO: many arguments may be redundant and could be inferred (move code from train_transformers?)
		# TODO: need to adapt for case where no labels are given (i.e. test set in the wild)
		has_labels = "met_type" in df.columns
		if has_labels:
			df["met_type"] = transform_met_types(df["met_type"], label_scheme=label_scheme)

		scheme_info = extract_scheme_info(label_scheme)
		primary_label_scheme = scheme_info["primary"]["name"]
		fallback_label = scheme_info["fallback_label"]
		iob2 = scheme_info["iob2"]

		contextualized_ex = create_examples(df,
											encoding_scheme=TAG2ID[primary_label_scheme],
											history_prev_sents=history_prev_sents,
											fallback_label=fallback_label,
											iob2=iob2)

		inputs, outputs, input_words = \
			contextualized_ex["input"], contextualized_ex["output"], contextualized_ex["input_words"]

		# [Original samples] are broken up into one or more [samples] for processing with the model
		num_samples = len(inputs)
		num_orig_samples = len(input_words)

		if isinstance(tokenizer_or_tokenizer_name, str):
			tokenizer = AutoTokenizer.from_pretrained(tokenizer_or_tokenizer_name)
		else:
			tokenizer = tokenizer_or_tokenizer_name

		enc_inputs = tokenizer(
			inputs, is_split_into_words=True,
			max_length=max_length, padding="max_length", truncation=True,
			return_overflowing_tokens=True, stride=stride,
			return_tensors="pt"
		)

		is_start_encountered = np.zeros(num_samples, dtype=bool)
		subsample_to_sample, subword_to_word = [], []
		enc_outputs = []
		for idx_ex, (curr_input_ids, idx_orig_ex) in enumerate(zip(enc_inputs["input_ids"],
																   enc_inputs["overflow_to_sample_mapping"])):
			subsample_to_sample.append(int(idx_orig_ex))
			curr_word_ids = enc_inputs.word_ids(idx_ex)
			curr_word_ids_shuf = []  # contains word IDs only for non-overlapping words, None for the rest

			# where does sequence actually start, i.e. after <bos>
			nonspecial_start = 0
			while curr_word_ids[nonspecial_start] is not None:
				nonspecial_start += 1

			# when an example is broken up, all but the first sub-example have first `stride` tokens overlapping with prev.
			ignore_n_overlapping = 0
			if is_start_encountered[idx_orig_ex]:
				ignore_n_overlapping = stride
			else:
				is_start_encountered[idx_orig_ex] = True

			fixed_out = []
			fixed_out += [LOSS_IGNORE_INDEX] * (nonspecial_start + ignore_n_overlapping)
			curr_word_ids_shuf.extend([None] * (nonspecial_start + ignore_n_overlapping))

			for idx_subw, w_id in enumerate(curr_word_ids[(nonspecial_start + ignore_n_overlapping):],
											start=(nonspecial_start + ignore_n_overlapping)):
				if curr_word_ids[idx_subw] is None:
					fixed_out.append(LOSS_IGNORE_INDEX)
					curr_word_ids_shuf.append(None)
				else:
					fixed_out.append(outputs[idx_orig_ex][w_id])
					# overlapping with some previous sample
					if outputs[idx_orig_ex][w_id] == LOSS_IGNORE_INDEX:
						curr_word_ids_shuf.append(None)
					else:
						curr_word_ids_shuf.append(w_id)

			enc_outputs.append(fixed_out)
			subword_to_word.append(curr_word_ids_shuf)

		enc_inputs["subsample_to_sample"] = subsample_to_sample
		enc_inputs["subword_to_word"] = subword_to_word
		enc_inputs["labels"] = torch.tensor(enc_outputs)
		del enc_inputs["overflow_to_sample_mapping"]

		train_dataset = TransformersTokenDataset(**enc_inputs)
		return train_dataset

	def align_word_predictions(self, preds, pad=False) -> List[int]:
		""" Converts potentially broken up and partially repeating (overlapping) predictions to word-level predictions."""
		converted_preds = [[None for _ in range(self.num_words_in_sample[idx_sample])]
						   for idx_sample in range(self.num_unique_samples)]

		subsample_to_sample = getattr(self, "subsample_to_sample")
		subword_to_word = getattr(self, "subword_to_word")
		for idx_subsample in range(preds.shape[0]):
			idx_sample = subsample_to_sample[idx_subsample]
			word_indices = subword_to_word[idx_subsample]
			curr_preds = preds[idx_subsample]

			assert len(word_indices) == len(curr_preds)
			for i, w_id in enumerate(word_indices):
				if w_id is None:
					continue

				# Prediction of first subword is assigned to be the prediction of word
				if converted_preds[idx_sample][w_id] is None:
					converted_preds[idx_sample][w_id] = curr_preds[i].item()

		converted_preds = [list(filter(lambda _w_id: _w_id is not None, _curr_preds)) for _curr_preds in converted_preds]

		if pad:
			max_words = max(map(lambda word_preds: len(word_preds), converted_preds))
			converted_preds = [_curr_preds + [LOSS_IGNORE_INDEX] * (max_words - len(_curr_preds))
							   for _curr_preds in converted_preds]

		return converted_preds


def extract_scheme_info(scheme_str: str):
	# scheme_str has the format "<label_type>[_<num_pos_labels>[_iob2]]", e.g. "independent_3_iob2"
	scheme_info = {
		"primary": {},
		"secondary": {},
		"fallback_label": None,
		"iob2": False
	}

	scheme_parts = scheme_str.split("_")
	# iob2 OFF
	if len(scheme_parts) == 1:
		scheme_name = scheme_parts[0]
		num_labels = len(TAG2ID[scheme_name])  # NOTE: this includes also the negative (fallback) label!
		scheme_info["primary"] = {"name": scheme_name, "num_pos_labels": num_labels - 1}
		scheme_info["secondary"] = {"name": scheme_name, "num_pos_labels": num_labels - 1}
		scheme_info["fallback_label"] = ID2TAG[scheme_name][FALLBACK_LABEL_INDEX]

	# iob2 OFF, number of provided labels indicates number of positive labels!
	elif len(scheme_parts) == 2:
		scheme_name, num_pos_labels = scheme_parts[0], int(scheme_parts[1])
		scheme_info["primary"] = {"name": scheme_name, "num_pos_labels": num_pos_labels}
		scheme_info["secondary"] = {"name": scheme_name, "num_pos_labels": num_pos_labels}
		scheme_info["fallback_label"] = ID2TAG[scheme_name][FALLBACK_LABEL_INDEX]

	# iob2 ON, number of provided labels indicates number of positive labels!
	elif len(scheme_parts) == 3:
		prim_scheme_name = f"{scheme_parts[0]}_iob2"
		sec_scheme_name = scheme_parts[0]
		num_labels = int(scheme_parts[1])

		scheme_info["primary"] = {"name": prim_scheme_name, "num_pos_labels": 2 * num_labels}  # B-, I- for each label
		scheme_info["secondary"] = {"name": sec_scheme_name, "num_pos_labels": num_labels}
		scheme_info["fallback_label"] = ID2TAG[prim_scheme_name][FALLBACK_LABEL_INDEX]
		scheme_info["iob2"] = True

	else:
		raise ValueError(f"Invalid format for scheme: '{scheme_str}'. Expecting format "
						 f"'<label_type>[_<num_pos_labels>[_iob2]]', where the brackets indicate optional parts.")

	return scheme_info


def transform_met_types(met_types: Iterable[Iterable[str]], label_scheme: str):
	""" Converts metaphore type using chosen label scheme. label_scheme determines which labels are kept as positive
	and how they are kept. See `TAG2ID` and `POS_MET_TYPES` in data.py to see priority.
	"""
	assert label_scheme in ["binary_1", "binary_2", "binary_3", "binary_4",
							"independent_1", "independent_2", "independent_3", "independent_4"]
	_, first_n = label_scheme.split("_")
	POS_LABELS_SET = set(POS_MET_TYPES[: int(first_n)])

	mapped_types = []
	if label_scheme.startswith("binary"):
		for curr_types in met_types:
			mapped_types.append(
				list(map(lambda _lbl: "metaphor" if _lbl in POS_LABELS_SET else "not_metaphor", curr_types))
			)
	elif label_scheme.startswith("independent"):
		for curr_types in met_types:
			mapped_types.append(
				list(map(lambda _lbl: _lbl if _lbl in POS_LABELS_SET else "O", curr_types))
			)
	else:
		raise NotImplementedError(f"Label scheme '{label_scheme}' unsupported")

	return mapped_types


def load_df(file_path) -> pd.DataFrame:
	""" Load data created using convert_komet.py. """
	df = pd.read_csv(file_path, sep="\t")
	if "sentence_words" in df.columns:
		df["sentence_words"] = df["sentence_words"].apply(ast.literal_eval)
	if "met_type" in df.columns:
		df["met_type"] = df["met_type"].apply(ast.literal_eval)
	if "met_frame" in df.columns:
		df["met_frame"] = df["met_frame"].apply(ast.literal_eval)

	return df


def create_examples(df: pd.DataFrame, encoding_scheme: Dict[str, int],
					history_prev_sents: int = 1,
					fallback_label: Optional[str] = "O",
					iob2: bool = False) -> Dict[str, List]:
	""" Creates examples (token inputs and token labels) out of data created using convert_komet.py.

	:param df:
	:param encoding_scheme: dict
	:param history_prev_sents: Number of previous sentences to take as context. Labels for this context are masked out
	so that the model does not predict token labels multiple times.
	:param fallback_label: Default negative label; used for unrecognized ("other") labels
	:param iob2: Encode labels as starting, inside, or outside (B-/I-/O)
	"""
	# TODO: this needs to be adapted for the case where no labels are given
	assert history_prev_sents >= 0
	examples_input = []
	examples_labels = []
	examples_input_words = []

	preprocess_labels = (lambda lbls: preprocess_iob2(lbls, fallback_label)) if iob2 else (lambda lbls: lbls)

	for curr_doc, curr_df in df.groupby("document_name"):
		sorted_order = np.argsort(curr_df["idx_sentence_glob"].values)
		curr_df_inorder = curr_df.iloc[sorted_order]

		tokens, labels = [], []
		for idx_row in range(len(sorted_order)):
			curr_row = curr_df_inorder.iloc[idx_row]
			tokens.append(curr_row["sentence_words"])
			labels.append(curr_row["met_type"])

			history_tokens = list(itertools.chain(*tokens[(-1 - history_prev_sents): -1]))
			curr_tokens = tokens[-1]

			curr_ex_tokens = history_tokens + curr_tokens

			unencoded_labels = preprocess_labels(labels[-1])
			curr_ex_labels = [-100] * len(history_tokens) + \
							 list(map(lambda str_lbl: encoding_scheme.get(str_lbl, encoding_scheme[fallback_label]), unencoded_labels))

			assert len(curr_ex_tokens) == len(curr_ex_labels)

			examples_input.append(curr_ex_tokens)
			examples_labels.append(curr_ex_labels)
			examples_input_words.append(curr_row["sentence_words"])

	return {
		"input": examples_input,
		"output": examples_labels,
		"input_words": examples_input_words
	}
