import argparse
import json
import os

import datasets
import numpy as np


def data_split(data):
	unique_documents = sorted(list(set(data["document_name"])))
	num_docs = len(unique_documents)
	doc_inds = np.random.permutation(num_docs)

	train_doc_inds = doc_inds[: int(0.7 * num_docs)]
	train_docs = set(unique_documents[_i] for _i in train_doc_inds)
	train_inds = [_i for _i in range(len(data)) if data.iloc[_i]["document_name"] in train_docs]

	dev_doc_inds = doc_inds[int(0.7 * num_docs): int(0.85 * num_docs)]
	dev_docs = set(unique_documents[_i] for _i in dev_doc_inds)
	dev_inds = [_i for _i in range(len(data)) if data.iloc[_i]["document_name"] in dev_docs]

	test_doc_inds = doc_inds[int(0.85 * num_docs):]
	test_docs = set(unique_documents[_i] for _i in test_doc_inds)
	test_inds = [_i for _i in range(len(data)) if data.iloc[_i]["document_name"] in test_docs]

	return train_inds, dev_inds, test_inds


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_handle", type=str, default="matejklemen/vuamc",
					choices=["cjvt/komet", "cjvt/gkomet", "cjvt/sloie", "cjvt/ssj500k",
							 "matejklemen/vuamc"])
parser.add_argument("--dataset_dir", type=str, default="vuamc")


if __name__ == "__main__":
	RAND_SEED = 321
	# RAND_SEED = 456  # used in vuamc because the initial seed produced an imbalanced split in terms of sentences
	args = parser.parse_args()
	np.random.seed(RAND_SEED)

	datasets_kwargs = {}
	if args.dataset_handle == "cjvt/ssj500k":
		datasets_kwargs["name"] = "multiword_expressions"
	data = datasets.load_dataset(args.dataset_handle)["train"]

	data = data.to_pandas()
	num_ex = data.shape[0]
	if args.dataset_handle in {"cjvt/komet", "cjvt/gkomet"}:
		data["sentence_words"] = data["sentence_words"].apply(lambda _curr_sent: _curr_sent.tolist())
		data["met_type"] = data["met_type"].apply(
			lambda _met_types: list(map(
				lambda _curr_type: {_k: _v.tolist() if _k == "word_indices" else _v for _k, _v in _curr_type.items()},
				_met_types.tolist()
			))
		)
		data["met_frame"] = data["met_frame"].apply(
			lambda _met_frames: list(map(
				lambda _curr_frame: {_k: _v.tolist() if _k == "word_indices" else _v for _k, _v in _curr_frame.items()},
				_met_frames.tolist()
			))
		)
	elif args.dataset_handle == "cjvt/sloie":
		data = data[["sentence_words", "is_idiom", "expression"]]

		data["sentence_words"] = data["sentence_words"].apply(lambda _curr_sent: _curr_sent.tolist())
		data["is_idiom"] = data["is_idiom"].apply(lambda npa: npa.tolist())
		data["document_name"] = [f"dummy{_idx}" for _idx in range(num_ex)]

		all_met_type, all_met_frame = [], []
		for idx_ex in range(num_ex):
			curr_ex = data.iloc[idx_ex]
			met_type, met_frame = None, "idiom"
			if "DA" in data.iloc[idx_ex]["is_idiom"]:
				met_type = "MRWi"
			elif "NEJASEN ZGLED" in curr_ex["is_idiom"]:
				met_type = "WIDLI"

			word_indices = [_i for _i in range(len(curr_ex["is_idiom"])) if curr_ex["is_idiom"][_i] not in {"*", "NE"}]

			if met_type is not None:
				all_met_type.append([{"type": met_type, "word_indices": word_indices}])
				all_met_frame.append([{"type": met_frame, "word_indices": word_indices}])
			else:
				all_met_type.append([])
				all_met_frame.append([])

		data["met_type"] = all_met_type
		data["met_frame"] = all_met_frame
	elif args.dataset_handle == "cjvt/ssj500k":
		data = data[["id_doc", "id_words", "words", "mwe_info"]]
		data = data.rename(columns={"words": "sentence_words", "id_doc": "document_name"})
		data["sentence_words"] = data["sentence_words"].apply(lambda _curr_sent: _curr_sent.tolist())
		data["mwe_info"] = data["mwe_info"].apply(
			lambda _met_types: list(map(
				lambda _curr_type: {_k: _v.tolist() if _k == "word_indices" else _v for _k, _v in _curr_type.items()},
				_met_types.tolist()
			))
		)

		MWE_MAPPING = {"ID": "idiom", "IRV": "inherently-reflexive-verb",
					   "LVC": "light verb construction", "LVC.full": "light verb construction", "LVC.cause": "light verb construction",
					   "VPC": "verb-particle-construction", "OTH": "other-verbal-MWE",
					   "IAV": "inherently-adpositional-verb", "VID": "idiom"}
		all_met_type, all_met_frame = [], []
		for idx_ex in range(num_ex):
			curr_ex = data.iloc[idx_ex]
			curr_type, curr_frame = [], []
			for mwe_info in curr_ex["mwe_info"]:
				curr_type.append({"type": "MRWi", "word_indices": mwe_info["word_indices"]})
				curr_frame.append({"type": MWE_MAPPING[mwe_info["type"]], "word_indices": mwe_info["word_indices"]})

			all_met_type.append(curr_type)
			all_met_frame.append(curr_frame)

		data["met_type"] = all_met_type
		data["met_frame"] = all_met_frame

	elif args.dataset_handle == "matejklemen/vuamc":
		data = data[["document_name", "words", "met_type"]]
		data = data.rename(columns={"words": "sentence_words"})
		data["sentence_words"] = data["sentence_words"].apply(lambda _curr_sent: _curr_sent.tolist())
		data["met_type"] = data["met_type"].apply(
			lambda _met_types: list(map(
				lambda _curr_type: {_k: _v.tolist() if _k == "word_indices" else _v for _k, _v in _curr_type.items()},
				_met_types.tolist()
			))
		)
		data["met_frame"] = [[] for _ in range(data.shape[0])]

		for idx_ex in range(data.shape[0]):
			curr_ex = data.iloc[idx_ex]

			idx_remap = {}
			for idx_word, word in enumerate(curr_ex["sentence_words"]):
				if len(word) != 0:
					idx_remap[idx_word] = len(idx_remap)

			# Others will be removed
			UNIFIED_MET_TYPES = {
				"mrw/met": "MRWi",
				"mrw/met/WIDLII": "WIDLI",
				"mrw/lit": "MRWd",
				"double": "MRWi",  # Note: could be a MRWd or MRWi, this is just a simplification
				"mFlag/lex": "MFlag",
				"mFlag/phrase": "MFlag",
				"mrw/impl": "MRWimp",
				"mrw/lit/WIDLII": "WIDLI",
				"mFlag/morph": "MFlag",
				"mFlag/lex/WIDLII": "WIDLI",
				"mFlag/phrase/WIDLII": "WIDLI",
				"mrw/impl/WIDLII": "WIDLI"
			}

			words, met_type = curr_ex[["sentence_words", "met_type"]].tolist()
			if len(idx_remap) != len(curr_ex["sentence_words"]):
				words = list(filter(lambda _word: len(_word) > 0, curr_ex["sentence_words"]))
				met_type = []

				for met_info in curr_ex["met_type"]:
					met_type.append({
						"type": met_info["type"],
						"word_indices": list(map(lambda _i: idx_remap[_i], met_info["word_indices"]))
					})
				data.at[idx_ex, "met_type"] = met_type

			met_type = []
			for met_info in curr_ex["met_type"]:
				if met_info["type"] not in UNIFIED_MET_TYPES:
					continue

				met_type.append({
					"type": UNIFIED_MET_TYPES[met_info["type"]],
					"word_indices": met_info["word_indices"]
				})
			data.at[idx_ex, "met_type"] = met_type
			data.at[idx_ex, "sentence_words"] = words

	train_inds, dev_inds, test_inds = data_split(data)
	train_data = data.iloc[train_inds]
	dev_data = data.iloc[dev_inds]
	test_data = data.iloc[test_inds]
	print(f"{train_data.shape[0]} train examples, "
		  f"{dev_data.shape[0]} dev examples, "
		  f"{test_data.shape[0]} test examples")

	if not os.path.exists(args.dataset_dir):
		os.makedirs(args.dataset_dir)

	with open(os.path.join(args.dataset_dir, "config.json"), "w", encoding="utf-8") as f:
		config = vars(args)
		config["random_seed"] = RAND_SEED
		json.dump(config, fp=f, indent=4)

	train_data.to_csv(os.path.join(args.dataset_dir, f"train_{args.dataset_dir}.tsv"), sep="\t", index=False)
	dev_data.to_csv(os.path.join(args.dataset_dir, f"dev_{args.dataset_dir}.tsv"), sep="\t", index=False)
	test_data.to_csv(os.path.join(args.dataset_dir, f"test_{args.dataset_dir}.tsv"), sep="\t", index=False)
