import logging
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter
from typing import List, Optional

import pandas as pd


def namespace(element):
	# https://stackoverflow.com/a/12946675
	m = re.match(r'\{.*\}', element.tag)
	return m.group(0) if m else ''


def resolve(element, expand_phrase: bool = False, only_noun_or_verb: bool = False, idioms_are_metaphors: bool = False):
	"""
		MRWd = direct metaphor
		MRWi = indirect metaphor
		WIDLI = borderline (cannot determine from context whether a word's metaphorical or basic meaning is intended)
		MFlag = metaphor signal
		MRWimp = MRWi, multipart? -> skip?
		* = ???
		Some tokens have multiple annotations??
	"""
	def _resolve_recursively(element, metaphor_type: str, frame_buffer: Optional[List]):
		# Leaf node: word or punctuation character
		if element.tag.endswith(("w", "pc")):
			if idioms_are_metaphors and metaphor_type == "idiom":
				metaphor_type = "MRWi"

			if element.tag.endswith("w"):
				pos_tag = element.attrib["ana"].split(":")[-1][0]  # e.g., ana="mte:Ncmpa" -> "N"
			else:
				pos_tag = "/"

			if len(frame_buffer) == 0:
				return element.text, metaphor_type, "O", pos_tag
			else:
				return element.text, metaphor_type, "/".join(frame_buffer), pos_tag
		# Annotated word or word group
		elif element.tag.endswith("seg"):
			mtype, new_frame_buffer = metaphor_type, frame_buffer
			if element.attrib["type"] == "frame":
				new_frame_buffer += [element.attrib["subtype"]]
			else:
				if element.attrib["subtype"] not in {"idiom", "MRWd", "MRWi", "WIDLI", "MFlag"}:
					logging.warning(f"Not covered: {element.attrib['subtype']}")
				# metaphor or idiom
				mtype = element.attrib["subtype"]

			parsed_data = []
			for child in element:
				if child.tag.endswith(("c", "vocal", "pause")):  # empty space betw. words or "special" word
					continue

				res = _resolve_recursively(child, mtype, new_frame_buffer)
				if isinstance(res, list):
					parsed_data.extend(res)
				else:
					parsed_data.append(res)

			return parsed_data

	curr_annotations = _resolve_recursively(element, "O", [])
	if curr_annotations is None:
		curr_annotations = []
	if not isinstance(curr_annotations, list):
		curr_annotations = [curr_annotations]

	if only_noun_or_verb:
		has_noun_or_verb = False
		for anno in curr_annotations:
			_, _, _, curr_pos = anno

			if curr_pos == "N" or curr_pos == "V":
				has_noun_or_verb = True
				break

		curr_annotations = list(map(lambda quad: quad[:3], curr_annotations))
		# If no noun or verb, mark as "not metaphor" even if the words constitute a metaphor
		if not has_noun_or_verb:
			processed_annotations = []
			for anno in curr_annotations:
				word, mtype, mframe = anno
				processed_annotations.append((word, "O", "O"))

			curr_annotations = processed_annotations

	# Expand metaphore type tag across the whole phrase (e.g., ["MRWd", "O"] -> ["MRWd", "MRWd"])
	if expand_phrase and len(curr_annotations) > 1:
		curr_mtypes = list(map(lambda trip: trip[1], curr_annotations))
		count_mtypes = Counter(curr_mtypes)
		most_common_mtype = "O"
		# Rules: prefer MRWi>MRWd>WIDLI>MFlag>O, no matter the tag frequency inside phrase
		for _mt in ["MRWi", "MRWd", "WIDLI", "MFlag"]:
			if _mt in count_mtypes:
				most_common_mtype = _mt
				break

		# Encode using IOB2
		curr_annotations = [(curr_tok, f"I-{most_common_mtype}", curr_frame) for (curr_tok, _, curr_frame) in curr_annotations]
		first_tok, _, first_frame = curr_annotations[0]
		curr_annotations[0] = (first_tok, f"B-{most_common_mtype}", first_frame)

	return curr_annotations


if __name__ == "__main__":
	logger = logging.basicConfig(level=logging.INFO)

	DATA_DIR = "/home/matej/Documents/data/g-komet"
	data_files = []  # TODO: sort data_files afterwards
	for fname in os.listdir(DATA_DIR):
		curr_path = os.path.join(DATA_DIR, fname)
		if os.path.isfile(curr_path) and fname.endswith(".xml") and fname != "G-Komet.xml":  # komet.xml = meta-file
			data_files.append(fname)

	data = {
		"document_name": [],
		"idx_paragraph": [],
		"idx_sentence": [],
		"idx_sentence_glob": [],
		"sentence_words": [],
		"met_type": [],
		"met_frame": []
	}
	label_counter = Counter()
	for fname in data_files:
		fpath = os.path.join(DATA_DIR, fname)
		print(fpath)
		curr_doc = ET.parse(fpath)
		root = curr_doc.getroot()
		NAMESPACE = namespace(root)

		idx_sent_glob = 0
		for idx_par, curr_par in enumerate(root.iterfind(f".//{NAMESPACE}p")):
			for idx_sent, curr_sent in enumerate(curr_par.iterfind(f"{NAMESPACE}s")):
				# print(curr_sent)
				words, types, frames = [], [], []
				for curr_el in curr_sent:
					if curr_el.tag.endswith(("w", "pc", "seg")):
						curr_res = resolve(curr_el)
						if not isinstance(curr_res, list):
							curr_res = [curr_res]
						for _el in curr_res:
							words.append(_el[0])
							types.append(_el[1])
							frames.append(_el[2])

				if len(words) == 0:
					continue

				data["document_name"].append(fname)
				data["idx_paragraph"].append(idx_par)
				data["idx_sentence"].append(idx_sent)
				data["idx_sentence_glob"].append(idx_sent_glob)
				data["sentence_words"].append(words)
				data["met_type"].append(types)
				data["met_frame"].append(frames)
				label_counter += Counter(types)

				idx_sent_glob += 1

	print("Label distribution:")
	print(label_counter)
	data = pd.DataFrame(data)
	data.to_csv("data.tsv", sep="\t", index=False)
