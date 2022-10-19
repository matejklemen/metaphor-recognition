import unittest
from typing import List

import pandas as pd
import torch
from transformers import AutoTokenizer

from data import Instance, EncodedInstance, create_examples, TransformersTokenDataset, EncodedSentenceInstance, \
	global_word_ids


class TestDataSpan(unittest.TestCase):
	def setUp(self):
		self.sample_df = pd.DataFrame({
			"document_name": ["dummy0", "dummy0", "dummy1", "dummy2"],
			"idx": [0, 1, 0, 0],
			"sentence_words": [
				["short", "sentence"],
				["short", "sentence", "with", "metaphors"],
				["a", "bit", "longer", "of", "a", "sentence"],
				["the", "longest", "of", "all", "the", "sentences", "in", "this", "example"]
			],
			"met_type": [
				[],
				[{"type": "MRWi", "word_indices": [3]}],
				[{"type": "MRWd", "word_indices": [1, 2]}, {"type": "MRWd", "word_indices": [5]}],
				[{"type": "MRWi", "word_indices": [0, 1]}, {"type": "MRWd", "word_indices": [3, 4, 5]}, {"type": "WIDLI", "word_indices": [7, 8]}]
			],
			"met_frame": [
				[],
				[],
				[],
				[]
			]
		})
		self.type_encoding = {"O": 0, "MRWi": 1, "MRWd": 2}
		self.tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)

	def test_create_instance(self):
		""" Test the creation of raw instance. """
		input_sent = ["short", "sentence", "with", "metaphors"]
		prev_sents = [["short", "sentence"]]
		met_type = [{"type": "MRWi", "word_indices": [3]}]
		met_frame = []

		# No history, no types, no frames
		instance = Instance(input_sent)
		self.assertEqual(len(instance.words), 4)
		self.assertEqual(len(instance.input_indices), 4)
		self.assertEqual(len(instance.sent_indices), 4)
		self.assertListEqual(instance.sent_indices, [0, 0, 0, 0])
		self.assertEqual(len(instance.history_indices), 0)
		self.assertListEqual(instance.met_type, [])
		self.assertListEqual(instance.met_frame, [])

		# No history, has types
		instance2 = Instance(input_sent, met_type=met_type)
		self.assertListEqual(instance2.met_type, [{"type": "MRWi", "word_indices_input": [3], "word_indices_instance": [3]}])

		# History, types and frames
		instance3 = Instance(input_sent, met_type=met_type, met_frame=met_frame, history_sents=prev_sents)
		self.assertEqual(len(instance3.words), 6)
		self.assertEqual(len(instance3.input_indices), 4)
		self.assertEqual(len(instance3.sent_indices), 6)
		self.assertListEqual(instance3.sent_indices, [0, 0, 1, 1, 1, 1])
		self.assertEqual(len(instance3.history_indices), 2)
		self.assertListEqual(instance3.met_type, [{"type": "MRWi", "word_indices_input": [3], "word_indices_instance": [5]}])
		self.assertListEqual(instance3.met_frame, [])

	def test_create_encoded_instance(self):
		""" Test the creation of encoded instance from a raw instance. """
		# "longest", "example" are misspelled to test handling of broken up sentences
		# NOTE: this example depends on the tokenizer == roberta-base
		input_sent = ["the", "longhest", "of", "all", "the", "sentences", "in", "this", "exzample"]
		prev_sents = [["short", "sentence"]]
		met_type = [{"type": "MRWi", "word_indices": [0, 1]}, {"type": "MRWd", "word_indices": [3, 4, 5]}, {"type": "WIDLI", "word_indices": [7, 8]}]
		met_frame = []

		instance = Instance(input_sent, met_type=met_type, met_frame=met_frame, history_sents=prev_sents)
		encoded_instances: List[EncodedInstance] = EncodedInstance.from_instance(instance, self.tokenizer,
																				 type_encoding=self.type_encoding,
																				 max_length=4, stride=1)

		self.assertEqual(len(encoded_instances), 13)
		self.assertEqual(len(encoded_instances[0].met_type), 0)
		# encoded_instance[1] in decoded form: ['<s>', ' sentence', ' the', '</s>']
		self.assertListEqual(encoded_instances[1].met_type,
							 [{"type": 1, "word_indices_input": [0], "word_indices_instance": [2], "subword_indices": [2]}])
		# encoded_instance[2] in decoded form:  ['<s>', ' the', ' long', '</s>']
		self.assertListEqual(encoded_instances[2].met_type,
							 [{"type": 1, "word_indices_input": [1], "word_indices_instance": [3], "subword_indices": [2]}])
		self.assertDictEqual(encoded_instances[2].word2subword, {3: [2]})

		# encoded_instance[3] in decoded form: ['<s>', ' long', 'hest', '</s>']
		self.assertListEqual(encoded_instances[3].met_type,
							 [{"type": 1, "word_indices_input": [1], "word_indices_instance": [3],
							   "subword_indices": [2]}])
		self.assertDictEqual(encoded_instances[3].word2subword, {3: [2]})

		# Make sure WIDLI was encoded as the fallback label ("O" -> 0) as it is not explicitly defined in type_encoding
		self.assertListEqual(encoded_instances[-1].met_type,
							 [{"type": 0, "word_indices_input": [8], "word_indices_instance": [10], "subword_indices": [2]}])

	def test_create_examples(self):
		created_examples = create_examples(self.sample_df, history_prev_sents=0)
		self.assertEqual(len(created_examples), 4)

		for created_inst in created_examples:
			self.assertEqual(len(created_inst.history_indices), 0)  # No previous sentences
			self.assertEqual(len(set(created_inst.sent_indices)), 1)  # All part of the same sentence

		created_examples = create_examples(self.sample_df, history_prev_sents=1)
		self.assertEqual(len(created_examples), 4)
		self.assertEqual(len(created_examples[0].history_indices), 0)  # First sentence has no previous sentence
		self.assertEqual(len(created_examples[1].history_indices), 2)

	def test_create_dataset_from_dataframe(self):
		# Case #1: no history, no breaking up of examples needed
		dataset = TransformersTokenDataset.from_dataframe(self.sample_df,
														  type_encoding=self.type_encoding,
														  max_length=32, stride=0, history_prev_sents=0,
														  tokenizer_or_tokenizer_name=self.tokenizer)

		self.assertEqual(len(dataset), 4)
		for attr_name in dataset.model_keys:
			self.assertListEqual(list(dataset.model_data[attr_name].shape), [4, 32])  # [num_examples, max_length]

		# Case #2: history and breaking up of examples needed, stride=0
		dataset = TransformersTokenDataset.from_dataframe(self.sample_df,
														  type_encoding=self.type_encoding,
														  max_length=8, stride=0, history_prev_sents=1,
														  tokenizer_or_tokenizer_name=self.tokenizer)
		self.assertEqual(len(dataset), 5)
		for attr_name in dataset.model_keys:
			self.assertListEqual(list(dataset.model_data[attr_name].shape), [5, 8])  # [num_examples, max_length]

		"""
			Instance, subwords: ['<s>', ' in', ' this', ' example', '</s>', '<pad>', '<pad>', '<pad>']
			Input (words): ['the', 'longest', 'of', 'all', 'the', 'sentences', 'in', 'this', 'example']
			Instance (words): ['the', 'longest', 'of', 'all', 'the', 'sentences', 'in', 'this', 'example']
		"""
		self.assertListEqual(dataset.enc_instances[4].met_type,
							 [{"type": 0, "word_indices_input": [7, 8], "word_indices_instance": [7, 8], "subword_indices": [2, 3]}])

		# Case #3: history and breaking up of examples needed, stride>0
		dataset = TransformersTokenDataset.from_dataframe(self.sample_df,
														  type_encoding=self.type_encoding,
														  max_length=10, stride=7, history_prev_sents=1,
														  tokenizer_or_tokenizer_name=self.tokenizer)

		self.assertEqual(len(dataset), 5)
		""" Check that metaphors are only tracked once across overlapping instances: ['all', 'the', 'sentences'] and 
		['this'] were already tracked in a previous instance.
		
		Instance, subwords: ['<s>', ' longest', ' of', ' all', ' the', ' sentences', ' in', ' this', ' example', '</s>']
		Input (words): ['the', 'longest', 'of', 'all', 'the', 'sentences', 'in', 'this', 'example']
		Instance (words): ['the', 'longest', 'of', 'all', 'the', 'sentences', 'in', 'this', 'example']
		"""
		self.assertListEqual(dataset.enc_instances[4].met_type,
							 [{"type": 0, "word_indices_input": [8], "word_indices_instance": [8],
							   "subword_indices": [8]}])

	def test_word_predictions(self):
		# "longest", "example" are misspelled to test handling of broken up sentences
		# NOTE: this example depends on the tokenizer == roberta-base
		input_sent = ["the", "longhest", "of", "all", "the", "sentences", "in", "this", "exzample"]
		prev_sents = [["short", "sentence"]]

		instance = Instance(input_sent, history_sents=prev_sents)
		encoded_instances: List[EncodedInstance] = EncodedInstance.from_instance(instance, self.tokenizer,
																				 type_encoding=self.type_encoding,
																				 max_length=8, stride=2)
		dataset = TransformersTokenDataset(encoded_instances)

		subw_preds = torch.tensor([
			# ['<s>', ' short', ' sentence', ' the', ' long', 'hest', ' of', '</s>']
			# Prediction for 'sentence' should be ignored because the sentence is part of the history
			# Prediction for 'longhest' should be 1 (prediction of first subword), not 2
			[0, 0, 1, 0, 1, 2, 0, 0],
			# ['<s>', 'hest', ' of', ' all', ' the', ' sentences', ' in', '</s>']
			# Prediction of subword 'hest' should be ignored because it is an overlapping part already taken into account
			[0, 3, 0, 0, 0, 1, 0, 0],
			# ['<s>', ' sentences', ' in', ' this', ' ex', 'z', 'ample', '</s>']
			[0, 2, 0, 0, 0, 1, 2, 0]
		])
		word_preds = dataset.word_predictions(subw_preds)
		self.assertListEqual(word_preds, [[0, 1, 0, 0, 0, 1, 0, 0, 0]])

	def test_word_predictions_aggregation(self):
		""" Test aggregation of subword predictions different from taking the first subword's prediction. """
		# "longest", "example" are misspelled to test handling of broken up sentences
		# NOTE: this example depends on the tokenizer == roberta-base
		input_sent = ["the", "longhest", "of", "all", "the", "sentences", "in", "this", "exzample"]
		prev_sents = [["short", "sentence"]]

		instance = Instance(input_sent, history_sents=prev_sents)
		encoded_instances: List[EncodedInstance] = EncodedInstance.from_instance(instance, self.tokenizer,
																				 type_encoding=self.type_encoding,
																				 max_length=8, stride=2)
		dataset = TransformersTokenDataset(encoded_instances)

		subw_preds = torch.tensor([
			# ['<s>', ' short', ' sentence', ' the', ' long', 'hest', ' of', '</s>']
			# Prediction for 'sentence' should be ignored because the sentence is part of the history
			# Prediction for 'longhest' should be 2; both 1 and 2 are equally frequent, so the earlier prediction should be taken
			[0, 0, 1, 0, 2, 1, 0, 0],
			# ['<s>', 'hest', ' of', ' all', ' the', ' sentences', ' in', '</s>']
			# Prediction of subword 'hest' should be ignored because it is an overlapping part already taken into account
			[0, 3, 0, 0, 0, 1, 0, 0],
			# ['<s>', ' sentences', ' in', ' this', ' ex', 'z', 'ample', '</s>']
			# exzample: 2x pred=2, 1x pred=0 => final_pred = 2
			[0, 2, 0, 0, 0, 2, 2, 0]
		])
		word_preds = dataset.word_predictions(subw_preds, aggr_strategy="majority")
		self.assertListEqual(word_preds, [[0, 2, 0, 0, 0, 1, 0, 0, 2]])

		subw_preds = torch.tensor([
			# ['<s>', ' short', ' sentence', ' the', ' long', 'hest', ' of', '</s>']
			# Prediction for 'sentence' should be ignored because the sentence is part of the history
			# Prediction for 'longhest' should be 2; both 1 and 2 are equally frequent, so the earlier prediction should be taken
			[0, 0, 1, 0, 2, 1, 0, 0],
			# ['<s>', 'hest', ' of', ' all', ' the', ' sentences', ' in', '</s>']
			# Prediction of subword 'hest' should be ignored because it is an overlapping part already taken into account
			[0, 3, 0, 0, 0, 1, 0, 0],
			# ['<s>', ' sentences', ' in', ' this', ' ex', 'z', 'ample', '</s>']
			# exzample: 2x pred=0, 1x pred=1 => final_pred = 1
			[0, 2, 0, 0, 0, 0, 1, 0]
		])
		word_preds = dataset.word_predictions(subw_preds, aggr_strategy="any")
		self.assertListEqual(word_preds, [[0, 2, 0, 0, 0, 1, 0, 0, 1]])

	def test_create_encoded_sentence_instance(self):
		# "longest", "example" are misspelled to test handling of broken up sentences
		# NOTE: this example depends on the tokenizer == roberta-base
		input_sent = ["the", "longhest", "of", "all", "the", "sentences", "in", "this", "exzample"]
		prev_sents = [["short", "sentence"]]
		met_type = [{"type": "MRWi", "word_indices": [0, 1]}, {"type": "MRWd", "word_indices": [3, 4, 5]},
					{"type": "WIDLI", "word_indices": [7, 8]}]
		met_frame = []

		# Case #1: no history
		instance = Instance(input_sent, met_type=met_type)
		# Make sure a warning is displayed because WIDLI is not handled by the provided type encoding
		with self.assertWarns(Warning):
			encoded_instance: EncodedSentenceInstance = EncodedSentenceInstance.from_instance(
				instance, self.tokenizer, type_encoding=self.type_encoding, max_length=8
			)

		# ['<s>', ' the', ' long', 'hest', ' of', ' all', ' the', '</s>']
		self.assertEqual(len(encoded_instance.subword2word), 6)
		# ["the", "longhest", "of", "all", "the"]
		self.assertEqual(len(encoded_instance.word2subword), 5)
		# [1x O, 1x MRWi, 1x MRWd]
		self.assertListEqual(encoded_instance.met_type, [1, 1, 1])
		# Make sure that the WIDLI metaphor is tracked, but that the subwords are not tracked as it gets truncated
		self.assertDictEqual(encoded_instance.met_type_metadata[-1],
							 {"type": 0, "word_indices_instance": [7, 8], "word_indices_input": [7, 8], "subword_indices": None})

		# Case #2: with history
		instance = Instance(input_sent, history_sents=prev_sents, met_type=met_type)
		with self.assertWarns(Warning):
			encoded_instance: EncodedSentenceInstance = EncodedSentenceInstance.from_instance(
				instance, self.tokenizer, type_encoding=self.type_encoding, max_length=8
			)

		# ['<s>', ' short', ' sentence', '</s>', '</s>', ' the', ' long', '</s>']
		self.assertListEqual(encoded_instance.met_type_metadata,
							 [
								# Make sure that only the subwords ["the", "long"] are tracked as "hest" gets truncated
								 {"type": 1, "word_indices_instance": [2, 3], "word_indices_input": [0, 1], "subword_indices": [5, 6]},
								 {"type": 2, "word_indices_instance": [5, 6, 7], "word_indices_input": [3, 4, 5], "subword_indices": None},
								 {"type": 0, "word_indices_instance": [9, 10], "word_indices_input": [7, 8], "subword_indices": None}
							 ])

	def test_global_word_ids(self):
		# Single sequence where IDs are already global
		self.assertListEqual(global_word_ids([None, 0, 1, 1, 2, None]),
							 [None, 0, 1, 1, 2, None])

		# A sequence pair, each with two words
		self.assertListEqual(global_word_ids([None, 0, 1, None, None, 0, 1, None]),
							 [None, 0, 1, None, None, 2, 3, None])
