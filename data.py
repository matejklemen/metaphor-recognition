import ast
import itertools
import warnings
from collections import Counter
from typing import Dict, Optional, List, Union, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

POS_MET_TYPES = ["MRWi", "MRWd", "WIDLI", "MFlag", "MRWimp", "bridge"]
TAG2ID = {
    "binary": {_tag: _i for _i, _tag in enumerate(["O"] + ["metaphor"])},
    "independent": {_tag: _i for _i, _tag in enumerate(["O"] + POS_MET_TYPES)}
}

ID2TAG = {curr_scheme: {_i: _tag for _tag, _i in TAG2ID[curr_scheme].items()} for curr_scheme in TAG2ID}
# This is used to mark labels that should not be taken into account in loss calculation
LOSS_IGNORE_INDEX = -100
# IMPORTANT: Assuming fallback label is encoded with 0 in every label scheme
FALLBACK_LABEL_INDEX = 0
FALLBACK_LABEL = "O"
PADDING_WORD_IDX = -1


def global_word_ids(local_word_ids: List[Union[int, None]]) -> List[Union[int, None]]:
    """ Postprocess word IDs to be global instead of local to each sequence.
    It is assumed that there is a separating token in between word IDs of multiple sequences!
    `local_word_ids` are the typically the word IDs as returned by HuggingFace transformers.

    E.g., turn [None, 0, 1, None, None, 0, 1, None] into [None, 0, 1, None, None, 2, 3, None].
    """
    postproc_word_ids = []
    prev_word_id = -1
    for _i in range(len(local_word_ids)):
        if local_word_ids[_i] is None:
            postproc_word_ids.append(local_word_ids[_i])
        else:
            if local_word_ids[_i] == local_word_ids[_i - 1]:
                postproc_word_ids.append(postproc_word_ids[-1])
            else:
                postproc_word_ids.append(prev_word_id + 1)
                prev_word_id = postproc_word_ids[-1]

    return postproc_word_ids


class Instance:
    def __init__(self, input_sent: List[str],
                 met_type: List[Dict] = None,
                 met_frame: List[Dict] = None,
                 history_sents: List[List[str]] = None):
        """ Represents an instance in its raw form, i.e. mostly in string format"""
        eff_met_type = met_type if met_type is not None else []
        eff_met_frame = met_frame if met_frame is not None else []
        eff_history_sents = history_sents if history_sents is not None else []

        self.words = []
        self.sent_indices = []
        # Words at which indices in `self.input_words` belong to either the history or the input sentence
        self.history_indices, self.input_indices = [], []

        for idx_sent, sent in enumerate(eff_history_sents):
            for i, word in enumerate(sent, start=len(self.words)):
                self.words.append(word)
                self.history_indices.append(i)
                self.sent_indices.append(idx_sent)

        idx_input_sent = (1 + self.sent_indices[-1]) if len(self.sent_indices) > 0 else 0
        for i, word in enumerate(input_sent, start=len(self.words)):
            self.words.append(word)
            self.input_indices.append(i)
            self.sent_indices.append(idx_input_sent)

        self.met_type = []
        for met_info in eff_met_type:
            self.met_type.append({
                "type": met_info["type"],
                "word_indices_input": met_info["word_indices"],
                "word_indices_instance": list(map(lambda idx: self.input_indices[idx], met_info["word_indices"]))
            })

        self.met_frame = []
        for frame_info in eff_met_frame:
            self.met_frame.append({
                "type": frame_info["type"],
                "word_indices_input": frame_info["word_indices"],
                "word_indices_instance": list(map(lambda idx: self.input_indices[idx], frame_info["word_indices"]))
            })

    def __str__(self):
        TRUNC_LEN = 5
        short_input = [self.words[idx_word] for idx_word in self.input_indices[:TRUNC_LEN]]
        short_input = "[{}{}]".format(', '.join(short_input), ", ..." if len(self.input_indices) > TRUNC_LEN else "")

        short_history = [self.words[idx_word] for idx_word in self.history_indices[:TRUNC_LEN]]
        short_history = "[{}{}]".format(', '.join(short_history), ", ..." if len(self.history_indices) > TRUNC_LEN else "")
        return f"Instance(input_sent={short_input}, history_sents={short_history})"


class EncodedInstance:
    def __init__(self, instance,
                 model_data: Dict,
                 word2subword: Dict[int, List[int]],
                 subword2word: Dict[int, int],
                 type_encoding: Dict[str, int],
                 enc_met_type: List[Dict] = None,
                 enc_met_frame: List[Dict] = None):
        self.instance = instance
        self.model_data = model_data

        self.subword2word = subword2word
        self.word2subword = word2subword
        self.type2id = type_encoding
        self.met_type = enc_met_type if enc_met_type is not None else []
        self.met_frame = enc_met_frame if enc_met_frame is not None else []

        self.next_sibling: Optional[EncodedInstance] = None  # To be set manually afterwards

    @staticmethod
    def from_instance(instance: Instance, tokenizer: PreTrainedTokenizer,
                      type_encoding: Dict[str, int], frame_encoding: Optional[Dict[str, int]] = None,
                      max_length: Optional[int] = 32, stride: Optional[int] = 0) -> List:
        encoded_words = tokenizer.encode_plus(instance.words, is_split_into_words=True,
                                              max_length=max_length, padding="max_length", truncation=True,
                                              return_overflowing_tokens=True, stride=stride, return_tensors="pt",
                                              return_special_tokens_mask=True)
        num_ex = len(encoded_words["input_ids"])

        return_instances = []
        is_start_encountered = False
        for idx_ex in range(num_ex):
            # Contains index of the word a subword belongs to, or None if subword is a special token
            word_ids: List[Union[int, None]] = encoded_words.word_ids(idx_ex)

            # Where does sequence actually start, i.e. after initial special tokens
            nonspecial_start = 0
            while word_ids[nonspecial_start] is None:
                nonspecial_start += 1

            # When an example is broken up, all but the first sub-example have first `stride` tokens overlapping with prev.
            ignore_n_overlapping = 0
            if is_start_encountered:
                ignore_n_overlapping = stride
            else:
                is_start_encountered = True

            # Contains word indices of subword belonging only to non-ovl. words (and vice versa for `word2subword`)
            subword2word: Dict[int, int] = {}
            word2subword: Dict[int, List[int]] = {}

            start_w_id = word_ids[nonspecial_start + ignore_n_overlapping]
            end_w_id = None
            for idx_subw, w_id in enumerate(word_ids[(nonspecial_start + ignore_n_overlapping):],
                                            start=(nonspecial_start + ignore_n_overlapping)):
                if w_id is not None:
                    subword2word[idx_subw] = w_id
                    existing_subw = word2subword.get(w_id, [])
                    existing_subw.append(idx_subw)
                    word2subword[w_id] = existing_subw

                    end_w_id = w_id

            proc_met_types = []
            # Only retain metaphors appearing in the current part of the original sample
            for met_info in instance.met_type:
                proc_met_info = {"type": type_encoding.get(met_info["type"], type_encoding[FALLBACK_LABEL]),
                                 "word_indices_input": [], "word_indices_instance": [],
                                 "subword_indices": []}

                for idx_word_input, idx_word_instance in zip(met_info["word_indices_input"],
                                                             met_info["word_indices_instance"]):
                    if start_w_id <= idx_word_instance <= end_w_id:
                        num_added_subw = 0
                        for idx_subw in word2subword[idx_word_instance]:
                            if subword2word.get(idx_subw, None) is None:
                                # This part of a metaphor was already taken into account in a previous sample,
                                # which this sample partially overlaps with => don't take it twice!
                                continue

                            proc_met_info["subword_indices"].append(idx_subw)
                            num_added_subw += 1

                        if num_added_subw > 0:
                            proc_met_info["word_indices_input"].append(idx_word_input)
                            proc_met_info["word_indices_instance"].append(idx_word_instance)

                # If at least part of a metaphor is present in the current instance, track it
                if len(proc_met_info["word_indices_instance"]) > 0:
                    proc_met_types.append(proc_met_info)

            # TODO: do the same for metaphor frames
            proc_met_frames = []
            # ...

            model_data = {_k: encoded_words[_k][idx_ex]
                          for _k in encoded_words.keys() if _k != "overflow_to_sample_mapping"}

            return_instances.append(
                EncodedInstance(instance, model_data, word2subword=word2subword, subword2word=subword2word,
                                type_encoding=type_encoding, enc_met_type=proc_met_types, enc_met_frame=proc_met_frames)
            )

        # Add order information for broken up instances
        for curr_inst, next_inst in zip(return_instances, return_instances[1:]):
            curr_inst.next_sibling = next_inst

        return return_instances


class EncodedSentenceInstance:
    def __init__(self, instance,
                 model_data: Dict,
                 word2subword: Dict[int, List[int]],
                 subword2word: Dict[int, int],
                 type_encoding: Dict[str, int],
                 enc_met_type: List[int] = None,
                 enc_met_frame: List[int] = None,
                 met_type_metadata: List[Dict] = None,
                 met_frame_metadata: List[Dict] = None):
        self.instance = instance
        self.model_data = model_data

        self.subword2word = subword2word
        self.word2subword = word2subword
        self.type2id = type_encoding
        self.met_type = enc_met_type if enc_met_type is not None else []
        self.met_frame = enc_met_frame if enc_met_frame is not None else []

        self.met_type_metadata = met_type_metadata if met_type_metadata is not None else {}
        self.met_frame_metadata = met_frame_metadata if met_frame_metadata is not None else {}

        self.next_sibling: Optional[EncodedInstance] = None  # To be set manually afterwards

    @staticmethod
    def from_instance(instance: Instance, tokenizer: PreTrainedTokenizer,
                      type_encoding: Dict[str, int], frame_encoding: Optional[Dict[str, int]] = None,
                      max_length: Optional[int] = 32):
        has_history = len(instance.history_indices) > 0
        if has_history:
            # pair of sequences: (history_sents, input_sent)
            input_or_input_pair = (
                [instance.words[_i] for _i in instance.history_indices],
                [instance.words[_i] for _i in instance.input_indices]
            )
        else:
            # single sequence: input_sent
            input_or_input_pair = instance.words

        encoded_data = tokenizer.encode_plus(input_or_input_pair, is_split_into_words=True, max_length=max_length,
                                             padding="max_length", truncation=True, return_tensors="pt",
                                             return_special_tokens_mask=True)

        subword2word: Dict[int, int] = {}
        word2subword: Dict[int, List[int]] = {}

        # Contains index of the word a subword belongs to, or None if subword is a special token
        word_ids: List[Union[int, None]] = encoded_data.word_ids()
        word_ids = global_word_ids(word_ids)

        # Where does sequence actually start, i.e. after initial special tokens
        nonspecial_start = 0
        while word_ids[nonspecial_start] is None:
            nonspecial_start += 1

        end_w_id = None
        for idx_subw, w_id in enumerate(word_ids[nonspecial_start:], start=nonspecial_start):
            if w_id is not None:
                subword2word[idx_subw] = w_id
                existing_subw = word2subword.get(w_id, [])
                existing_subw.append(idx_subw)
                word2subword[w_id] = existing_subw
                end_w_id = w_id

        met_type_count = [0 for _ in range(len(type_encoding))]
        proc_met_types = []
        for met_info in instance.met_type:
            met_type_count[type_encoding.get(met_info["type"], type_encoding[FALLBACK_LABEL])] += 1
            proc_met_info = {"type": type_encoding.get(met_info["type"], type_encoding[FALLBACK_LABEL]),
                             "word_indices_input": [], "word_indices_instance": [],
                             "subword_indices": []}

            for idx_word_input, idx_word_instance in zip(met_info["word_indices_input"],
                                                         met_info["word_indices_instance"]):
                # For non-truncated words, keep track of their subword indices
                if idx_word_instance <= end_w_id:
                    for idx_subw in word2subword[idx_word_instance]:
                        proc_met_info["subword_indices"].append(idx_subw)

                proc_met_info["word_indices_input"].append(idx_word_input)
                proc_met_info["word_indices_instance"].append(idx_word_instance)

            # If a metaphor was truncated, we still want to track it, but don't connect it to any subwords
            if len(proc_met_info["subword_indices"]) == 0:
                proc_met_info["subword_indices"] = None

            proc_met_types.append(proc_met_info)

        # If no types are annotated or if there are no types, mark it as a non-metaphoric sentence
        if sum(met_type_count[1:]) == 0:
            met_type_count[FALLBACK_LABEL_INDEX] = 1

        if met_type_count[FALLBACK_LABEL_INDEX] > 1 or \
                (met_type_count[FALLBACK_LABEL_INDEX] == 1 and sum(met_type_count[1:]) > 0):
            warnings.warn("At least one metaphor type could not be encoded with the provided type mapping, so it is "
                          f"being counted as '{FALLBACK_LABEL}'. To fix this, either remap the type or provide an "
                          f"additional ID for the type in `type_encoding`.", Warning)

        # TODO: do the same for metaphor frames, below are just sensible defaults
        # ...
        met_frame_count = [1]  # sentence with no frames
        proc_met_frames = []

        # For consistency with EncodedInstance, remove the batch dimension from tensors
        model_data = {_k: _v[0] for _k, _v in encoded_data.items()}
        return EncodedSentenceInstance(instance, model_data,
                                       word2subword=word2subword, subword2word=subword2word,
                                       type_encoding=type_encoding,
                                       enc_met_type=met_type_count, enc_met_frame=met_frame_count,
                                       met_type_metadata=proc_met_types, met_frame_metadata=proc_met_frames)


class TransformersTokenDataset(Dataset):
    def __init__(self, encoded_instances: List[EncodedInstance], **kwargs):
        self.enc_instances = encoded_instances

        assert len(self.enc_instances) > 0, "Cannot construct dataset without encoded instances"
        self.model_keys = list(self.enc_instances[0].model_data.keys())
        self.model_data = {attr: [] for attr in self.model_keys}

        for idx_ex in range(len(encoded_instances)):
            for attr in self.model_keys:
                self.model_data[attr].append(encoded_instances[idx_ex].model_data[attr])

        for attr in self.model_keys:
            self.model_data[attr] = torch.stack(self.model_data[attr])

        self.target_names = kwargs.get("target_names", ["met_type", "met_frame"])
        self.target_data = {target_name: [] for target_name in self.target_names}
        for target_name in self.target_names:
            self.target_data[target_name] = list(map(lambda _enc_inst: getattr(_enc_inst, target_name),
                                                     encoded_instances))

        # Represents the alignment of encoded instances to an external data source (e.g., a dataframe)
        self.alignment_indices = None

    @property
    def input_sentences(self):
        seen_instances = set()
        sents: List[List[str]] = []
        for enc in self.enc_instances:
            if enc.instance not in seen_instances:
                sents.append([enc.instance.words[_i] for _i in enc.instance.input_indices])
            seen_instances.add(enc.instance)

        return sents

    def __getitem__(self, item):
        ret_dict = {k: self.model_data[k][item] for k in self.model_keys}
        ret_dict["indices"] = item
        return ret_dict

    def targets(self, item):
        ret_dict = {target_name: [] for target_name in self.target_names}
        for target_name in self.target_names:
            for _i in item:
                ret_dict[target_name].append(self.target_data[target_name][_i])

        return ret_dict

    def __len__(self):
        return len(self.enc_instances)

    @staticmethod
    def from_dataframe(dataframe, type_encoding: Dict[str, int], max_length, stride=0, history_prev_sents=0,
                       tokenizer_or_tokenizer_name: Union[str, AutoTokenizer] = "EMBEDDIA/sloberta",
                       frame_encoding: Dict[str, int] = None):
        # TODO: introduce `max_met_length`: padding, truncation of met_type and met_frame (improve training efficiency!)
        if isinstance(tokenizer_or_tokenizer_name, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_or_tokenizer_name)
        else:
            tokenizer = tokenizer_or_tokenizer_name

        alignment_inds, instances = create_examples(dataframe, history_prev_sents=history_prev_sents)
        encoded_instances = []

        num_ex = len(instances)
        for idx_ex in range(num_ex):
            curr_inst = instances[idx_ex]
            curr_enc_insts = EncodedInstance.from_instance(curr_inst, tokenizer, type_encoding=type_encoding,
                                                           max_length=max_length, stride=stride)
            encoded_instances.extend(curr_enc_insts)

        dataset = TransformersTokenDataset(encoded_instances)
        dataset.alignment_indices = alignment_inds
        return dataset

    def word_predictions(self, preds: torch.Tensor, pad: Optional[bool] = False,
                         aggr_strategy: Optional[str] = "first") -> List[List[int]]:
        """ Converts potentially broken up and partially repeating (overlapping) predictions to word-level predictions."""
        assert preds.shape[0] == len(self.enc_instances), "Predictions need to be provided for each encoded instance " \
                                                          f"({preds.shape[0]} predictions != {len(self.enc_instances)} encoded instances)"

        if aggr_strategy == "first":
            aggr_fn = lambda subw_preds: subw_preds[0]
        elif aggr_strategy == "majority":
            # Prediction of word = most common prediction of subwords; tie-break: order of appearance
            aggr_fn = lambda subw_preds: Counter(subw_preds).most_common(n=1)[0][0]
        elif aggr_strategy == "any":
            # Prediction of word =
            #   - if at least one subword is different from C=0, predict most common among those;
            #   - otherwise, predict 0
            def aggr_fn(subw_preds):
                c = Counter(subw_preds)
                for _pred, _count in c.most_common():
                    if _pred != FALLBACK_LABEL_INDEX:
                        return _pred

                return FALLBACK_LABEL_INDEX
        else:
            raise NotImplementedError(aggr_strategy)

        converted_preds: List[List] = []
        inst2idx: Dict[Instance, int] = {}  # instance (obj) -> position of instance (in `converted_preds`)
        for idx_enc, enc_instance in enumerate(self.enc_instances):
            if enc_instance.instance not in inst2idx:
                inst2idx[enc_instance.instance] = len(converted_preds)
                converted_preds.append(
                    [[] for _ in range(len(enc_instance.instance.words))]
                )

            idx_inst = inst2idx[enc_instance.instance]
            for idx_subw, pred in enumerate(preds[idx_enc]):
                # `idx_word` will be None for special and overlapping tokens
                idx_word = enc_instance.subword2word.get(idx_subw, None)
                if idx_word is None:
                    continue

                converted_preds[idx_inst][idx_word].append(int(pred))

        idx2inst: Dict[int, Instance] = {_i: _inst for _inst, _i in inst2idx.items()}

        postproc_preds: List[List[int]] = []
        # Only keep predictions for the input sentence (i.e. drop for history), aggregate subword into word predictions
        for idx_inst, preds in enumerate(converted_preds):
            input_preds = [aggr_fn(preds[_i]) for _i in idx2inst[idx_inst].input_indices]
            postproc_preds.append(input_preds)

        if pad:
            max_words = max(map(lambda word_preds: len(word_preds), postproc_preds))
            postproc_preds = [_curr_preds + [LOSS_IGNORE_INDEX] * (max_words - len(_curr_preds))
                               for _curr_preds in postproc_preds]

        return postproc_preds


class TransformersSentenceDataset(Dataset):
    def __init__(self, encoded_instances: List[EncodedSentenceInstance], **kwargs):
        self.enc_instances = encoded_instances

        assert len(self.enc_instances) > 0, "Cannot construct dataset without encoded instances"
        self.model_keys = list(self.enc_instances[0].model_data.keys())
        self.model_data = {attr: [] for attr in self.model_keys}

        for attr in self.model_keys:
            for idx_ex in range(len(encoded_instances)):
                self.model_data[attr].append(encoded_instances[idx_ex].model_data[attr])
            self.model_data[attr] = torch.stack(self.model_data[attr])

        self.target_names = kwargs.get("target_names", ["met_type", "met_frame"])
        self.target_data = {
            target_name: torch.tensor(
                list(map(lambda _enc_inst: getattr(_enc_inst, target_name), encoded_instances))
            ) for target_name in self.target_names
        }

        # Represents the alignment of encoded instances to an external data source (e.g., a dataframe)
        self.alignment_indices = None

    @property
    def input_sentences(self):
        # As opposed to TransformersTokenDataset.input_sentences, each EncodedSentenceInstance contains exactly
        # one input sentence. Note that these input sentences are not necessarily equal to what the model gets as
        # input as the sentence may get truncated depending on the provided history sentences and `max_length`
        sents: List[List[str]] = []
        for enc in self.enc_instances:
            sents.append([enc.instance.words[_i] for _i in enc.instance.input_indices])

        return sents

    def __getitem__(self, item):
        ret_dict = {k: self.model_data[k][item] for k in self.model_keys}
        ret_dict["indices"] = item
        return ret_dict

    def targets(self, item):
        return {target_name: self.target_data[target_name][item] for target_name in self.target_names}

    def __len__(self):
        return len(self.enc_instances)

    @staticmethod
    def from_dataframe(dataframe, type_encoding: Dict[str, int], max_length, stride=0, history_prev_sents=0,
                       tokenizer_or_tokenizer_name: Union[str, AutoTokenizer] = "EMBEDDIA/sloberta",
                       frame_encoding: Dict[str, int] = None):
        # TODO: introduce `max_met_length`: padding, truncation of met_type and met_frame (improve training efficiency!)
        if isinstance(tokenizer_or_tokenizer_name, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_or_tokenizer_name)
        else:
            tokenizer = tokenizer_or_tokenizer_name

        alignment_inds, instances = create_examples(dataframe, history_prev_sents=history_prev_sents)
        encoded_instances = []

        num_ex = len(instances)
        for idx_ex in range(num_ex):
            curr_inst = instances[idx_ex]
            curr_enc_inst = EncodedSentenceInstance.from_instance(curr_inst, tokenizer,
                                                                  type_encoding=type_encoding,max_length=max_length)
            encoded_instances.append(curr_enc_inst)

        dataset = TransformersSentenceDataset(encoded_instances)
        dataset.alignment_indices = alignment_inds
        return dataset


def create_examples(df: pd.DataFrame, history_prev_sents: int = 0) -> Tuple[List[int], List[Instance]]:
    assert history_prev_sents >= 0

    instances, encoded_instances = [], []
    has_types = "met_type" in df.columns
    has_frames = "met_frame" in df.columns
    has_sentence_indices = "idx" in df.columns
    alignment_indices = []

    for curr_doc, curr_df in df.groupby("document_name"):
        alignment_indices.append(curr_df.index.tolist())
        if has_sentence_indices:
            sorted_order = np.argsort(curr_df["idx"].values)
        else:
            # If no indices are given, assume pre-sorted order
            sorted_order = np.arange(curr_df.shape[0])

        curr_df_inorder = curr_df.iloc[sorted_order]

        for idx_row in range(len(sorted_order)):
            curr_row = curr_df_inorder.iloc[idx_row]
            history_context = []
            if history_prev_sents > 0:
                history_context = [curr_df_inorder.iloc[_i]["sentence_words"]
                                   for _i in range(max(0, idx_row - history_prev_sents), idx_row)]

            instances.append(Instance(input_sent=curr_row["sentence_words"],
                                      met_type=curr_row["met_type"] if has_types else None,
                                      met_frame=curr_row["met_frame"] if has_frames else None,
                                      history_sents=history_context))

    return list(itertools.chain(*alignment_indices)), instances


def data_split(data):
    unique_documents = sorted(list(set(data["document_name"])))
    num_docs = len(unique_documents)
    doc_inds = np.random.permutation(num_docs)

    train_doc_inds = doc_inds[: int(0.7 * num_docs)]
    train_docs = set(unique_documents[_i] for _i in train_doc_inds)
    train_inds = [_i for _i in range(len(data)) if data[_i]["document_name"] in train_docs]

    dev_doc_inds = doc_inds[int(0.7 * num_docs): int(0.85 * num_docs)]
    dev_docs = set(unique_documents[_i] for _i in dev_doc_inds)
    dev_inds = [_i for _i in range(len(data)) if data[_i]["document_name"] in dev_docs]

    test_doc_inds = doc_inds[int(0.85 * num_docs):]
    test_docs = set(unique_documents[_i] for _i in test_doc_inds)
    test_inds = [_i for _i in range(len(data)) if data[_i]["document_name"] in test_docs]

    return train_inds, dev_inds, test_inds


def load_df(file_path) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep="\t")
    if "sentence_words" in df.columns:
        df["sentence_words"] = df["sentence_words"].apply(ast.literal_eval)
    if "met_type" in df.columns:
        df["met_type"] = df["met_type"].apply(ast.literal_eval)
    if "met_frame" in df.columns:
        df["met_frame"] = df["met_frame"].apply(ast.literal_eval)
    if "preds_transformed" in df.columns:
        try:
            df["preds_transformed"] = df["preds_transformed"].apply(ast.literal_eval)
        except ValueError:
            print("Skipping conversion of 'preds_transformed' - ast.literal_eval could not be applied")
    if "true_transformed" in df.columns:
        try:
            df["true_transformed"] = df["true_transformed"].apply(ast.literal_eval)
        except ValueError:
            print("Skipping conversion of 'true_transformed' - ast.literal_eval could not be applied")

    return df


if __name__ == "__main__":
    # from data import load_df
    # df = load_df("/home/matej/Documents/metaphor-detection/data/komet_hf_format/train_komet_hf_format.tsv")
    # data = TransformersTokenDataset.from_dataframe(df, type_encoding=TAG2ID["independent"],
    #                                                max_length=32, stride=16, history_prev_sents=0,
    #                                                tokenizer_or_tokenizer_name="EMBEDDIA/sloberta")

    import datasets
    np.random.seed(321)
    data = datasets.load_dataset("cjvt/komet")["train"]

    train_inds, dev_inds, test_inds = data_split(data)
    data = data.to_pandas()
    train_data = data.iloc[train_inds]
    dev_data = data.iloc[dev_inds]
    test_data = data.iloc[test_inds]
