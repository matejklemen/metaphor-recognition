import argparse
import json
import logging
import os
import sys
from typing import List, Dict

import numpy as np
import torch
from torch import optim, nn
from tqdm import trange
from transformers import AutoModelForTokenClassification, AutoTokenizer

from custom_modules import TokenClassifierWithLayerCombination
from data import FALLBACK_LABEL_INDEX, load_df, create_examples, TransformersTokenDataset
from utils import token_f1, token_precision, token_recall, visualize_predictions

MAX_THRESH_TO_CHECK = 100


def extract_pred_data(pred_probas: torch.Tensor,
                      correct_types: List[List[Dict]],
                      special_tokens_mask: torch.Tensor):
    free_tokens_mask = torch.logical_not(special_tokens_mask.bool())
    _pred_logproba, _correct_class = [], []

    # Positive examples = annotated metaphors
    for idx_ex in range(pred_probas.shape[0]):
        for met_info in correct_types[idx_ex]:
            free_tokens_mask[idx_ex, met_info["subword_indices"]] = False
            _pred_logproba.append(torch.log(
                torch.mean(pred_probas[[idx_ex], met_info["subword_indices"]], dim=0)
            ))
            _correct_class.append(met_info["type"])

    # TODO: num_types is taken from outside scope - DANGEROUS
    # TODO: device is taken from outside scope - DANGEROUS
    _pred_logproba = torch.stack(_pred_logproba) if len(_pred_logproba) > 0 else torch.zeros((0, num_types)).to(DEVICE)
    # Negative examples = all other tokens
    _pred_logproba = torch.cat((_pred_logproba,
                               torch.log(pred_probas[free_tokens_mask])))
    _correct_class.extend([FALLBACK_LABEL_INDEX] * (_pred_logproba.shape[0] - len(_correct_class)))

    return _pred_logproba, torch.tensor(_correct_class)


parser = argparse.ArgumentParser()
parser.add_argument("--experiment_dir", type=str, default="debug_token_modeling")
parser.add_argument("--train_path", type=str,
                    default="/home/matej/Documents/metaphor-detection/data/komet/train.tsv")
parser.add_argument("--dev_path", type=str,
                    default="/home/matej/Documents/metaphor-detection/data/komet/dev.tsv")
parser.add_argument("--history_prev_sents", type=int, default=0)
parser.add_argument("--type_scheme", type=str, default="binary", choices=["binary"])
parser.add_argument("--mrwi", action="store_true")
parser.add_argument("--mrwd", action="store_true")
parser.add_argument("--mflag", action="store_true")
parser.add_argument("--widli", action="store_true")
parser.add_argument("--mrwimp", action="store_true")
parser.add_argument("--bridge", action="store_true")

parser.add_argument("--pretrained_name_or_path", type=str, default="EMBEDDIA/sloberta")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--validate_every_n_examples", type=int, default=3000)
parser.add_argument("--early_stopping_rounds", type=int, default=5)
parser.add_argument("--validation_metric", type=str, default="f1_score_binary",
                    choices=["f1_score_binary"])
parser.add_argument("--optimize_bin_threshold", action="store_true",
                    help="If set and '--type_scheme' is binary, optimize the decision threshold on the validation set "
                         "using the best model")
parser.add_argument("--word_prediction_strategy", type=str, default="first",
                    choices=["first", "majority", "any"])
parser.add_argument("--layer_combination", action="store_true")
parser.add_argument("--tune_last_only", action="store_true")

parser.add_argument("--random_seed", type=int, default=17)
parser.add_argument("--use_cpu", action="store_true")


if __name__ == "__main__":
    args = parser.parse_args()
    if os.path.exists(args.experiment_dir):
        raise ValueError("experiment_dir exists, so the data could accidentally get overriden."
                         f"Please remove '{args.experiment_dir}' manually and rerun the script")

    os.makedirs(args.experiment_dir, exist_ok=True)

    # Set up logging to file and stdout
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for curr_handler in [logging.StreamHandler(sys.stdout),
                         logging.FileHandler(os.path.join(args.experiment_dir, f"train.log"))]:
        curr_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s"))
        logger.addHandler(curr_handler)

    if not torch.cuda.is_available() and not args.use_cpu:
        args.use_cpu = True

    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)

    for k, v in vars(args).items():
        v_str = str(v)
        v_str = f"...{v_str[-(50 - 3):]}" if len(v_str) > 50 else v_str
        logging.info(f"|{k:30s}|{v_str:50s}|")

    DEVICE = torch.device("cpu") if args.use_cpu else torch.device("cuda")
    DEV_BATCH_SIZE = args.batch_size * 2

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

    best_thresh = None
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_name_or_path)
    model = model_class.from_pretrained(args.pretrained_name_or_path, num_labels=num_types)
    model = model.to(DEVICE)

    if args.tune_last_only:
        assert not args.layer_combination, \
            "--tune_last_only is unimplemented for case where --layer_combination is set"

        logging.info("Freezing all but the last (linear classifier) layer")
        for name, param in model.named_parameters():
            if name not in {"classifier.weight", "classifier.bias"}:
                param.requires_grad = False

    optimizer = optim.AdamW(params=model.parameters(), lr=args.learning_rate)

    tokenizer.save_pretrained(args.experiment_dir)
    model.save_pretrained(args.experiment_dir)

    df_train = load_df(args.train_path)
    # Remap valid metaphor types according to the used scheme (e.g., binary metaphors),
    # keeping track only of positive metaphor spans
    for idx_ex in range(df_train.shape[0]):
        new_met_type = []
        for met_info in df_train.iloc[idx_ex]["met_type"]:
            new_type = MET_TYPE_MAPPING.get(met_info["type"], "O")
            if new_type != "O":
                met_info["type"] = new_type
                new_met_type.append(met_info)

        df_train.at[idx_ex, "met_type"] = new_met_type

    df_dev = load_df(args.dev_path)
    for idx_ex in range(df_dev.shape[0]):
        new_met_type = []
        for met_info in df_dev.iloc[idx_ex]["met_type"]:
            new_type = MET_TYPE_MAPPING.get(met_info["type"], "O")
            if new_type != "O":
                met_info["type"] = new_type
                new_met_type.append(met_info)

        df_dev.at[idx_ex, "met_type"] = new_met_type

    # Set max_length automatically to 99th percentile of training lengths
    _, _train_instances = create_examples(df_train, history_prev_sents=args.history_prev_sents)
    train_lengths = sorted([len(_curr)
                            for _curr in
                            tokenizer.batch_encode_plus(list(map(lambda _inst: _inst.words, _train_instances)),
                                                        is_split_into_words=True)["input_ids"]])
    max_length = train_lengths[int(0.99 * len(train_lengths))]
    args.max_length = max_length
    args.stride = args.max_length // 2
    logging.info(f"Automatically determined max_length={args.max_length} and stride={args.stride}")

    train_set = TransformersTokenDataset.from_dataframe(df_train, history_prev_sents=args.history_prev_sents,
                                                        type_encoding=type_encoding, frame_encoding=frame_encoding,
                                                        max_length=args.max_length, stride=args.stride,
                                                        tokenizer_or_tokenizer_name=tokenizer)
    dev_set = TransformersTokenDataset.from_dataframe(df_dev, history_prev_sents=args.history_prev_sents,
                                                      type_encoding=type_encoding, frame_encoding=frame_encoding,
                                                      max_length=args.max_length, stride=args.stride,
                                                      tokenizer_or_tokenizer_name=tokenizer)

    logging.info(f"Loaded {len(train_set)} training instances, {len(dev_set)} validation instances")

    dev_true_dense = torch.zeros_like(dev_set.model_data["input_ids"])  # zeros_like because 0 == fallback index
    expected_types = dev_set.target_data["met_type"]
    for idx_ex in range(dev_true_dense.shape[0]):
        for met_info in expected_types[idx_ex]:
            dev_true_dense[idx_ex, met_info["subword_indices"]] = met_info["type"]

    dev_true_word_padded: np.ndarray = np.array(dev_set.word_predictions(dev_true_dense, pad=True))
    dev_true_word_unpadded: List[List[int]] = dev_set.word_predictions(dev_true_dense, pad=False)

    loss_fn = nn.NLLLoss()
    validation_fn = lambda y_true, y_pred: 0.0  # placeholder
    if args.validation_metric == "f1_score_binary":
        validation_fn = lambda y_true, y_pred: token_f1(true_labels=y_true, pred_labels=y_pred, pos_label=1)
    else:
        raise NotImplementedError(args.validation_metric)

    num_subsets = (len(train_set) + args.validate_every_n_examples - 1) // args.validate_every_n_examples
    best_dev_metric, no_increase = 0.0, 0

    for idx_epoch in range(args.num_epochs):
        logging.info(f"Epoch {1 + idx_epoch}/{args.num_epochs}")
        train_loss, nb = 0.0, 0
        shuf_indices = torch.randperm(len(train_set))

        for idx_subset in range(num_subsets):
            # 1. Training step
            model.train()
            s_sub, e_sub = idx_subset * args.validate_every_n_examples, \
                           (idx_subset + 1) * args.validate_every_n_examples
            sub_indices = shuf_indices[s_sub: e_sub]
            num_tr_batches = (sub_indices.shape[0] + args.batch_size - 1) // args.batch_size

            for idx_batch in trange(num_tr_batches):
                s_b, e_b = idx_batch * args.batch_size, (idx_batch + 1) * args.batch_size
                curr_batch = train_set[sub_indices[s_b: e_b]]

                indices = curr_batch["indices"]
                curr_batch_device = {_k: _v.to(DEVICE) for _k, _v in curr_batch.items()
                                     if _k not in {"indices", "special_tokens_mask"}}
                ground_truth = train_set.targets(indices)
                expected_types = ground_truth["met_type"]

                probas = torch.softmax(model(**curr_batch_device)["logits"], dim=-1)
                pred_logproba, correct_class = extract_pred_data(probas, expected_types,
                                                                 curr_batch["special_tokens_mask"].to(DEVICE))
                pred_logproba = pred_logproba.to(DEVICE)
                correct_class = correct_class.to(DEVICE)

                loss = loss_fn(pred_logproba, correct_class)
                train_loss += float(loss.cpu())
                nb += 1

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            logging.info(f"[Subset #{1 + idx_subset}/{num_subsets}] Train loss: {train_loss / max(1, nb):.4f}")

            if sub_indices.shape[0] < (args.validate_every_n_examples // 2):
                logging.info(f"\tSKIPPING validation because current training subset was too small "
                             f"({sub_indices.shape[0]} < {args.validate_every_n_examples // 2} examples)")
                continue

            # 2. Validation step
            dev_preds_dense = []
            with torch.inference_mode():
                model.eval()

                num_dev_batches = (len(dev_set) + DEV_BATCH_SIZE - 1) // DEV_BATCH_SIZE
                dev_indices = torch.arange(len(dev_set))

                for idx_batch in trange(num_dev_batches):
                    s_b, e_b = idx_batch * DEV_BATCH_SIZE, (idx_batch + 1) * DEV_BATCH_SIZE
                    batch_indices = dev_indices[s_b: e_b]
                    curr_batch = dev_set[batch_indices]

                    indices = curr_batch["indices"]
                    curr_batch_device = {_k: _v.to(DEVICE) for _k, _v in curr_batch.items()
                                         if _k not in {"indices", "special_tokens_mask"}}

                    probas = torch.softmax(model(**curr_batch_device)["logits"], dim=-1)

                    # Dev set: group predictions at word-level to ensure comparability across different models
                    _dev_preds = torch.argmax(probas, dim=-1).cpu()
                    dev_preds_dense.append(_dev_preds)

            dev_preds_dense = torch.cat(dev_preds_dense)
            dev_preds_word = dev_set.word_predictions(dev_preds_dense, pad=True,
                                                      aggr_strategy=args.word_prediction_strategy)

            dev_metric = validation_fn(y_true=dev_true_word_padded, y_pred=np.array(dev_preds_word))
            logging.info(f"\tValidation {args.validation_metric}: {dev_metric:.4f}")

            if dev_metric > best_dev_metric:
                logging.info(f"\t\tNew best, saving checkpoint!")
                model.save_pretrained(args.experiment_dir)
                best_dev_metric, no_increase = dev_metric, 0
            else:
                no_increase += 1

            if no_increase == args.early_stopping_rounds:
                logging.info(f"Stopping early because the validation metric did not improve "
                             f"for {args.early_stopping_rounds} rounds")
                break

        if no_increase == args.early_stopping_rounds:
            break

    logging.info(f"Best validation {args.validation_metric}: {best_dev_metric:.4f}")
    logging.info("Reloading best model for evaluation...")
    model = model_class.from_pretrained(args.experiment_dir).to(DEVICE)

    if args.layer_combination:
        logging.info(f"Layer combination parameters:")
        logging.info(model.layer_combination.data.cpu().flatten())

    # Obtain predictions on validation set with the best model, save them, visualize them
    with torch.inference_mode():
        model.eval()

        num_dev_batches = (len(dev_set) + DEV_BATCH_SIZE - 1) // DEV_BATCH_SIZE
        dev_indices = torch.arange(len(dev_set))
        dev_probas_dense = []

        for idx_batch in trange(num_dev_batches):
            s_b, e_b = idx_batch * DEV_BATCH_SIZE, (idx_batch + 1) * DEV_BATCH_SIZE
            batch_indices = dev_indices[s_b: e_b]
            curr_batch = dev_set[batch_indices]

            indices = curr_batch["indices"]
            curr_batch_device = {_k: _v.to(DEVICE) for _k, _v in curr_batch.items()
                                 if _k not in {"indices", "special_tokens_mask"}}

            probas = torch.softmax(model(**curr_batch_device)["logits"], dim=-1)
            dev_probas_dense.append(probas.cpu())

    dev_probas_dense = torch.cat(dev_probas_dense)
    if args.optimize_bin_threshold:
        thresh_to_check = sorted(list(set(torch.flatten(dev_probas_dense[:, :, 1]).tolist())))
        if len(thresh_to_check) > MAX_THRESH_TO_CHECK:
            thresh_to_check = [thresh_to_check[int((idx / 100) * len(thresh_to_check))]
                               for idx in range(1, 100)]

        best_thresh, metric_with_best_thresh = None, 0.0
        for curr_thresh in thresh_to_check:
            dev_preds_dense = (dev_probas_dense[:, :, 1] >= curr_thresh).int()
            dev_preds_word_padded = np.array(dev_set.word_predictions(dev_preds_dense, pad=True,
                                                                      aggr_strategy=args.word_prediction_strategy))

            curr_metric = validation_fn(y_true=dev_true_word_padded, y_pred=dev_preds_word_padded)
            if curr_metric > metric_with_best_thresh:
                best_thresh = curr_thresh
                metric_with_best_thresh = curr_metric

        logging.info(f"[Threshold optimization] Best T={best_thresh}, validation {args.validation_metric} = {metric_with_best_thresh:.4f}")
        args.decision_threshold_bin = best_thresh
        dev_preds_dense = (dev_probas_dense[:, :, 1] >= best_thresh).int()
    else:
        dev_preds_dense = torch.argmax(dev_probas_dense, dim=-1)

    with open(os.path.join(args.experiment_dir, "experiment_config.json"), "w") as f:
        json.dump(vars(args), fp=f, indent=4)

    dev_preds_word_padded = dev_set.word_predictions(dev_preds_dense, pad=True,
                                                     aggr_strategy=args.word_prediction_strategy)
    dev_preds_word_unpadded = list(
        map(lambda instance_types: list(map(lambda _idx_type: rev_type_encoding[_idx_type], instance_types)),
            dev_set.word_predictions(dev_preds_dense, pad=False,
                                     aggr_strategy=args.word_prediction_strategy))
    )

    dev_true_word_unpadded: List[List[str]] = list(
        map(lambda instance_types: list(map(lambda _idx_type: rev_type_encoding[_idx_type], instance_types)),
            dev_true_word_unpadded)
    )

    final_dev_p = token_precision(true_labels=dev_true_word_padded,
                                  pred_labels=np.array(dev_preds_word_padded))
    final_dev_r = token_recall(true_labels=dev_true_word_padded,
                               pred_labels=np.array(dev_preds_word_padded))
    final_dev_f1 = token_f1(true_labels=dev_true_word_padded,
                            pred_labels=np.array(dev_preds_word_padded))

    logging.info(f"Dev metrics using best model: P = {final_dev_p:.4f}, R = {final_dev_r:.4f}, F1 = {final_dev_f1:.4f}")
    with open(os.path.join(args.experiment_dir, "pred_visualization.html"), "w", encoding="utf-8") as f:
        dev_words = [dev_set.input_sentences[_i] for _i in dev_set.alignment_indices]
        dev_preds_word_unpadded = [dev_preds_word_unpadded[_i] for _i in dev_set.alignment_indices]
        dev_true_word_unpadded = [dev_true_word_unpadded[_i] for _i in dev_set.alignment_indices]

        visualization_html = visualize_predictions(dev_words,
                                                   ytoken_pred=dev_preds_word_unpadded, ytoken_true=dev_true_word_unpadded)
        print(visualization_html, file=f)

    df_dev["preds_transformed"] = dev_preds_word_unpadded
    df_dev["true_transformed"] = dev_true_word_unpadded
    df_dev.to_csv(os.path.join(args.experiment_dir, "dev_results.tsv"), index=False, sep="\t")
