import logging
import os
import re
import xml.etree.ElementTree as ET
from typing import List, Optional

import pandas as pd


def namespace(element):
	# https://stackoverflow.com/a/12946675
	m = re.match(r'\{.*\}', element.tag)
	return m.group(0) if m else ''


def resolve(element):
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
			if len(frame_buffer) == 0:
				return element.text, metaphor_type, "O"
			else:
				return element.text, metaphor_type, "/".join(frame_buffer)
		# Annotated word or word group
		elif element.tag.endswith("seg"):
			mtype, new_frame_buffer = "O", frame_buffer
			if element.attrib["subtype"] != "frame":
				if element.attrib["subtype"] not in {"MRWd", "MRWi", "WIDLI", "MFlag"}:
					logging.warning(f"Not covered: {element.attrib['subtype']}")

				mtype = element.attrib["subtype"]
			else:
				new_frame_buffer += [element.attrib["ana"]]

			parsed_data = []
			for child in element:
				if child.tag.endswith("c"):  # spaces between words, skip
					continue
				res = _resolve_recursively(child, mtype, new_frame_buffer)
				if isinstance(res, list):
					parsed_data.extend(res)
				else:
					parsed_data.append(res)

			return parsed_data

	return _resolve_recursively(element, "O", [])


if __name__ == "__main__":
	logger = logging.basicConfig(level=logging.INFO)

	DATA_DIR = "/home/matej/Documents/data/komet"
	data_files = []
	for fname in os.listdir(DATA_DIR):
		curr_path = os.path.join(DATA_DIR, fname)
		if os.path.isfile(curr_path) and fname.endswith(".xml") and fname != "komet.xml":  # komet.xml = meta-file
			data_files.append(fname)

	data = {
		"document_name": [],
		"idx_paragraph": [],
		"idx_sentence": [],
		"sentence_words": [],
		"met_type": [],
		"met_frame": []
	}
	for fname in data_files:
		fpath = os.path.join(DATA_DIR, fname)
		print(fpath)
		curr_doc = ET.parse(fpath)
		root = curr_doc.getroot()
		NAMESPACE = namespace(root)

		for idx_par, curr_par in enumerate(root.iterfind(f"{NAMESPACE}p")):
			for idx_sent, curr_sent in enumerate(curr_par.iterfind(f"{NAMESPACE}s")):
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

				data["document_name"].append(fname)
				data["idx_paragraph"].append(idx_par)
				data["idx_sentence"].append(idx_sent)
				data["sentence_words"].append(words)
				data["met_type"].append(types)
				data["met_frame"].append(frames)

	data = pd.DataFrame(data)
	data.to_csv("data.tsv", sep="\t", index=False)
