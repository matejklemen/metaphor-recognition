import argparse
import json
import logging
import os
import sys
import time
from copy import deepcopy
from typing import List, Optional

import numpy as np
import torch
from tqdm import trange
from transformers import AutoModelForTokenClassification, AutoTokenizer

from custom_modules import TokenClassifierWithLayerCombination
from data import load_df, TransformersTokenDataset
from utils import token_f1, token_recall, token_precision, visualize_predictions

MAX_THRESH_TO_CHECK = 100

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_dir", type=str, default="debug_token_modeling")
parser.add_argument("--data_path", type=str, default="data/komet-traindev/dev.tsv")
parser.add_argument("--history_prev_sents", type=int, default=0)

parser.add_argument("--pretrained_name_or_path", type=str, default="EMBEDDIA/sloberta")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--decision_threshold_bin", type=float, default=None,
                    help="Specify a decision threshold to be used in binary classification of metaphors")
parser.add_argument("--word_prediction_strategy", type=str, default="first",
                    choices=["first", "majority", "any"])
parser.add_argument("--layer_combination", action="store_true")
parser.add_argument("--mcd_rounds", type=int, default=None)

parser.add_argument("--random_seed", type=int, default=17)
parser.add_argument("--use_cpu", action="store_true")


if __name__ == "__main__":
    args = parser.parse_args()
    curr_ts = int(time.time())
    if os.path.exists(args.experiment_dir):
        args.experiment_dir = f"{args.experiment_dir}_{curr_ts}"
        print(f"Warning: --experiment_dir exists, so it is implicitly changed to {args.experiment_dir}")

    os.makedirs(args.experiment_dir, exist_ok=True)

    # Set up logging to file and stdout
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for curr_handler in [logging.StreamHandler(sys.stdout),
                         logging.FileHandler(os.path.join(args.experiment_dir, f"predict.log"))]:
        curr_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s"))
        logger.addHandler(curr_handler)

    # Sensible defaults
    existing_config = {
        "type_scheme": "binary",
        "mrwi": True, "mrwd": True, "mflag": False, "widli": False, "mrwimp": True, "bridge": False,
        "history_prev_sents": 0, "max_length": 64, "stride": 0, "word_prediction_strategy": "first",
        "layer_combination": False
    }

    if os.path.exists(args.pretrained_name_or_path):
        logging.info("Loading existing config information from experiment_config.json")
        with open(os.path.join(args.pretrained_name_or_path, "experiment_config.json"), "r") as f:
            existing_config = json.load(f)

        # Override some arguments
        for override_key in ["experiment_dir", "data_path", "pretrained_name_or_path", "use_cpu", "random_seed",
                             "batch_size", "decision_threshold_bin", "word_prediction_strategy", "layer_combination"]:
            if getattr(args, override_key) is not None:
                existing_config[override_key] = getattr(args, override_key)

    for attr, attr_val in existing_config.items():
        setattr(args, attr, attr_val)

    if not torch.cuda.is_available() and not args.use_cpu:
        logging.info("Implicitly setting --use_cpu because no CUDA device was found on the system")
        args.use_cpu = True

    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)

    for k, v in vars(args).items():
        v_str = str(v)
        v_str = f"...{v_str[-(50 - 3):]}" if len(v_str) > 50 else v_str
        logging.info(f"|{k:30s}|{v_str:50s}|")

    DEVICE = torch.device("cpu") if args.use_cpu else torch.device("cuda")

    valid_types = []
    if args.mrwi:
        valid_types.append("MRWi")
    if args.mrwd:
        valid_types.append("MRWd")
    if args.mflag:
        valid_types.append("MFlag")
    if args.widli:
        valid_types.append("WIDLI")
    if args.mrwimp:
        valid_types.append("MRWimp")
    if args.bridge:
        valid_types.append("bridge")
    if len(valid_types) == 0:
        raise ValueError("No valid types were specified, please set one of --mrwi, --mrwd, --mflag, --widli, "
                         "--mrwimp, or --bridge")

    if args.type_scheme == "binary":
        MET_TYPE_MAPPING = {_type: "metaphor" for _type in valid_types}  # everything else maps to "O"
        type_encoding = {"O": 0, "metaphor": 1}
    else:
        raise NotImplementedError(args.type_scheme)

    rev_type_encoding = {_idx: _type for _type, _idx in type_encoding.items()}
    num_types = len(type_encoding)
    frame_encoding = None  # TODO: implement me

    model_class = TokenClassifierWithLayerCombination if args.layer_combination else AutoModelForTokenClassification
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_name_or_path)
    model = model_class.from_pretrained(args.pretrained_name_or_path).to(DEVICE)

    df_test = load_df(args.data_path)
    has_type = "met_type" in df_test.columns

    if has_type:
        # Transform the metaphor annotations according to the scheme and drop uninteresting metaphor types
        for idx_ex in range(df_test.shape[0]):
            new_met_type = []
            for met_info in df_test.iloc[idx_ex]["met_type"]:
                new_type = MET_TYPE_MAPPING.get(met_info["type"], "O")
                if new_type != "O":
                    met_info_copy = deepcopy(met_info)
                    met_info_copy["type"] = new_type
                    new_met_type.append(met_info_copy)

            df_test.at[idx_ex, "met_type"] = new_met_type

    test_set = TransformersTokenDataset.from_dataframe(df_test, history_prev_sents=args.history_prev_sents,
                                                       type_encoding=type_encoding, frame_encoding=frame_encoding,
                                                       max_length=args.max_length, stride=args.stride,
                                                       tokenizer_or_tokenizer_name=tokenizer)

    logging.info(f"Loaded {len(test_set)} test instances")

    test_true_word_padded, test_true_word_unpadded = None, None
    if has_type:
        test_true_dense = torch.zeros_like(test_set.model_data["input_ids"])  # zeros_like because 0 == fallback index
        expected_types = test_set.target_data["met_type"]
        for idx_ex in range(test_true_dense.shape[0]):
            for met_info in expected_types[idx_ex]:
                test_true_dense[idx_ex, met_info["subword_indices"]] = met_info["type"]

        test_true_word_padded: Optional[np.ndarray] = np.array(test_set.word_predictions(test_true_dense, pad=True))
        test_true_word_unpadded: Optional[List[List[int]]] = test_set.word_predictions(test_true_dense, pad=False)

    num_rounds = int(args.mcd_rounds) if args.mcd_rounds is not None else 1
    # Obtain predictions on TEST set with the best model, save them, visualize them
    with torch.inference_mode():
        if num_rounds > 1:
            model.train()
        else:
            model.eval()

        num_test_batches = (len(test_set) + args.batch_size - 1) // args.batch_size
        test_indices = torch.arange(len(test_set))
        test_probas = []

        for idx_round in range(num_rounds):
            logging.info(f"Prediction round {1 + idx_round}/{num_rounds}")
            round_probas = []

            for idx_batch in trange(num_test_batches):
                s_b, e_b = idx_batch * args.batch_size, (idx_batch + 1) * args.batch_size
                batch_indices = test_indices[s_b: e_b]
                curr_batch = test_set[batch_indices]

                indices = curr_batch["indices"]
                curr_batch_device = {_k: _v.to(DEVICE) for _k, _v in curr_batch.items()
                                     if _k not in {"indices", "special_tokens_mask"}}

                probas = torch.softmax(model(**curr_batch_device)["logits"], dim=-1)
                round_probas.append(probas.cpu())

            test_probas.append(torch.cat(round_probas))

        test_probas = torch.stack(test_probas)  # [num_rounds, len(test_set), max_length, num_types]
        mean_probas = torch.mean(test_probas, dim=0)
        if num_rounds < 2:
            sd_probas = torch.zeros_like(mean_probas)
        else:
            sd_probas = torch.std(test_probas, dim=0)

        df_test["mean_proba"] = mean_probas[test_set.alignment_indices].tolist()
        df_test["sd_proba"] = sd_probas[test_set.alignment_indices].tolist()

    if args.type_scheme == "binary" and args.decision_threshold_bin is not None:
        logging.info(f"Using manually specified decision threshold T={args.decision_threshold_bin}")
        test_preds_dense = (mean_probas[:, :, 1] >= args.decision_threshold_bin).int()
    else:
        test_preds_dense = torch.argmax(mean_probas, dim=-1)

    test_preds_word_padded = test_set.word_predictions(test_preds_dense, pad=True,
                                                       aggr_strategy=args.word_prediction_strategy)
    test_preds_word_unpadded = list(
        map(lambda instance_types: list(map(lambda _idx_type: rev_type_encoding[_idx_type], instance_types)),
            test_set.word_predictions(test_preds_dense, pad=False,
                                      aggr_strategy=args.word_prediction_strategy))
    )

    if has_type:
        import matplotlib.pyplot as plt
        thresh_to_check = sorted(list(set(torch.flatten(mean_probas[:, :, 1]).tolist())))
        if len(thresh_to_check) > MAX_THRESH_TO_CHECK:
            thresh_to_check = [thresh_to_check[int((idx / 100) * len(thresh_to_check))]
                               for idx in range(1, 100)]

        # Holds (<thresh>, P, R, F1) for each threshold
        thresh_stats = []
        for curr_thresh in thresh_to_check:
            _test_preds_dense = (mean_probas[:, :, 1] >= curr_thresh).int().numpy()
            _test_preds_word_padded = np.array(test_set.word_predictions(_test_preds_dense, pad=True,
                                                                         aggr_strategy=args.word_prediction_strategy))
            thresh_stats.append((curr_thresh,
                                 token_precision(true_labels=test_true_word_padded,
                                                 pred_labels=_test_preds_word_padded),
                                 token_recall(true_labels=test_true_word_padded,
                                              pred_labels=_test_preds_word_padded),
                                 token_f1(true_labels=test_true_word_padded,
                                          pred_labels=_test_preds_word_padded)))

        thresh_stats = sorted(thresh_stats, key=lambda curr_stats: curr_stats[0])
        plt.title("F1 score at different thresholds")
        plt.plot(thresh_stats, list(map(lambda curr_stats: curr_stats[3], thresh_stats)), color="blue",
                 linestyle="None")
        plt.xlim([0.0, 1.0 + 0.01])
        plt.ylim([0.0, 1.0 + 0.01])
        plt.xlabel("Threshold")
        plt.ylabel("F1")
        plt.savefig(os.path.join(args.experiment_dir, "f1_curve.png"))

        logging.info("**********************")
        prec_sorted_stats = sorted(thresh_stats, key=lambda curr_stats: (curr_stats[1], curr_stats[2]))
        _T, _P, _R, _F1 = prec_sorted_stats[-1]
        logging.info(f"[Maximize token_precision] T = {_T}, validation P = {_P:.4f}, R = {_R:.4f}, F1 = {_F1:.4f}")

        rec_sorted_stats = sorted(thresh_stats, key=lambda curr_stats: (curr_stats[2], curr_stats[1]))
        _T, _P, _R, _F1 = rec_sorted_stats[-1]
        logging.info(f"[Maximize token_recall] T = {_T}, validation P = {_P:.4f}, R = {_R:.4f}, F1 = {_F1:.4f}")

        f1_sorted_stats = sorted(thresh_stats, key=lambda curr_stats: (curr_stats[3], curr_stats[0]))
        _T, _P, _R, _F1 = f1_sorted_stats[-1]
        logging.info(f"[Maximize token_F1] T = {_T}, validation P = {_P:.4f}, R = {_R:.4f}, F1 = {_F1:.4f}")
        logging.info("IMPORTANT: This maximization is not automatically taken into account in the final test metrics")
        logging.info("**********************")

        # Convert encoded classes to strings
        test_true_word_unpadded = list(
            map(lambda instance_types: list(map(lambda _idx_type: rev_type_encoding[_idx_type], instance_types)),
                test_true_word_unpadded)
        )

        final_test_p = token_precision(true_labels=test_true_word_padded,
                                       pred_labels=np.array(test_preds_word_padded))
        final_test_r = token_recall(true_labels=test_true_word_padded,
                                    pred_labels=np.array(test_preds_word_padded))
        final_test_f1 = token_f1(true_labels=test_true_word_padded,
                                 pred_labels=np.array(test_preds_word_padded))
        logging.info(f"Test metrics using model: "
                     f"P = {final_test_p:.4f}, R = {final_test_r:.4f}, F1 = {final_test_f1:.4f}")
    else:
        logging.info("Skipping test set evaluation because no 'met_type' column is provided in the test file")

    with open(os.path.join(args.experiment_dir, "pred_visualization_test.html"), "w", encoding="utf-8") as f:
        # Align examples and their predictions to the order in the source dataframe
        test_words = [test_set.input_sentences[_i] for _i in test_set.alignment_indices]
        test_preds_word_unpadded = [test_preds_word_unpadded[_i] for _i in test_set.alignment_indices]
        if has_type:
            test_true_word_unpadded = [test_true_word_unpadded[_i] for _i in test_set.alignment_indices]

        visualization_html = visualize_predictions(test_words,
                                                   ytoken_pred=test_preds_word_unpadded, ytoken_true=test_true_word_unpadded)
        print(visualization_html, file=f)

    df_test["preds_transformed"] = test_preds_word_unpadded
    if has_type:
        df_test["true_transformed"] = test_true_word_unpadded

    df_test.to_csv(os.path.join(args.experiment_dir, "test_results.tsv"), index=False, sep="\t")
