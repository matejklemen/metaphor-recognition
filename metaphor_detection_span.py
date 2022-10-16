import argparse
import json
import logging
import os
import sys
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from tqdm import trange
from transformers import AutoTokenizer, AutoModelForTokenClassification

from data_span import TransformersTokenDataset, load_df


def extract_pred_data(pred_probas: torch.Tensor,
                      correct_types: List[List[Dict]],
                      special_tokens_mask: torch.Tensor):
    free_tokens_mask = torch.logical_not(special_tokens_mask.bool())
    pred_logproba, correct_class = [], []

    # Positive examples = annotated metaphors
    for idx_ex in range(pred_probas.shape[0]):
        for met_info in correct_types[idx_ex]:
            free_tokens_mask[idx_ex, met_info["subword_indices"]] = False
            pred_logproba.append(torch.log(
                torch.mean(pred_probas[[idx_ex], met_info["subword_indices"]], dim=0)
            ))
            correct_class.append(met_info["type"])

    pred_logproba = torch.stack(pred_logproba) if len(pred_logproba) > 0 else torch.zeros((0, num_types)).to(DEVICE)
    # Negative examples = all other tokens
    pred_logproba = torch.cat((pred_logproba,
                               torch.log(pred_probas[free_tokens_mask])))
    correct_class.extend([0] * (pred_logproba.shape[0] - len(correct_class)))  # TODO: FALLBACK_IDX?

    return pred_logproba, torch.tensor(correct_class)


parser = argparse.ArgumentParser()
parser.add_argument("--experiment_dir", type=str, default="debug_span_modeling")
parser.add_argument("--train_path", type=str,
                    default="/home/matej/Documents/metaphor-detection/data/komet_hf_format/train_komet_hf_format.tsv")
parser.add_argument("--dev_path", type=str,
                    default="/home/matej/Documents/metaphor-detection/data/komet_hf_format/dev_komet_hf_format.tsv")
parser.add_argument("--history_prev_sents", type=int, default=0)

parser.add_argument("--pretrained_name_or_path", type=str, default="EMBEDDIA/sloberta")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--max_length", type=int, default=32)
parser.add_argument("--stride", type=int, default=None)
parser.add_argument("--validate_every_n_examples", type=int, default=3000)
parser.add_argument("--early_stopping_rounds", type=int, default=5)
parser.add_argument("--validation_metric", type=str, default="f1_score_binary",
                    choices=["f1_score_binary"])

parser.add_argument("--random_seed", type=int, default=17)
parser.add_argument("--use_cpu", action="store_true")

if __name__ == "__main__":
    args = parser.parse_args()
    if args.stride is None:
        args.stride = args.max_length // 2

    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)

    if not torch.cuda.is_available() and not args.use_cpu:
        args.use_cpu = True

    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)

    for k, v in vars(args).items():
        v_str = str(v)
        v_str = f"...{v_str[-(50 - 3):]}" if len(v_str) > 50 else v_str
        logging.info(f"|{k:30s}|{v_str:50s}|")

    with open(os.path.join(args.experiment_dir, "experiment_config.json"), "w") as f:
        json.dump(vars(args), fp=f, indent=4)

    # Set up logging to file and stdout
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for curr_handler in [logging.StreamHandler(sys.stdout),
                         logging.FileHandler(os.path.join(args.experiment_dir, "experiment.log"))]:
        curr_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s"))
        logger.addHandler(curr_handler)

    DEVICE = torch.device("cpu") if args.use_cpu else torch.device("cuda")
    DEV_BATCH_SIZE = args.batch_size * 2

    # TODO: wire me with argparse
    MET_TYPE_MAPPING = {"MRWi": "metaphor", "MRWd": "metaphor"}  # everything else maps to "O"
    type_encoding = {"O": 0, "metaphor": 1}
    num_types = len(type_encoding)
    frame_encoding = None  # TODO: implement me

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_name_or_path)
    model = AutoModelForTokenClassification.from_pretrained(
        args.pretrained_name_or_path, num_labels=num_types
    ).to(DEVICE)
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

    train_set = TransformersTokenDataset.from_dataframe(df_train, history_prev_sents=args.history_prev_sents,
                                                        type_encoding=type_encoding, frame_encoding=frame_encoding,
                                                        max_length=args.max_length, stride=args.stride,
                                                        tokenizer_or_tokenizer_name=tokenizer)
    dev_set = TransformersTokenDataset.from_dataframe(df_dev, history_prev_sents=args.history_prev_sents,
                                                      type_encoding=type_encoding, frame_encoding=frame_encoding,
                                                      max_length=args.max_length, stride=args.stride,
                                                      tokenizer_or_tokenizer_name=tokenizer)

    logging.info(f"{len(train_set)} training instances, {len(dev_set)} validation instances")

    loss_fn = nn.NLLLoss()
    validation_fn = lambda y_true, y_pred: 0.0  # placeholder
    if args.validation_metric == "f1_score_binary":
        validation_fn = lambda y_true, y_pred: f1_score(y_true=y_true, y_pred=y_pred, average="binary", pos_label=1)
    else:
        raise NotImplementedError(args.validation_metric)

    num_subsets = (len(train_set) + args.validate_every_n_examples - 1) // args.validate_every_n_examples
    best_dev_metric, no_increase = 0.0, 0

    for idx_epoch in range(args.num_epochs):
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
            with torch.inference_mode():
                model.eval()

                dev_preds, dev_true = [], []
                num_dev_batches = (len(dev_set) + DEV_BATCH_SIZE - 1) // DEV_BATCH_SIZE
                dev_indices = torch.arange(len(dev_set))
                for idx_batch in trange(num_dev_batches):
                    s_b, e_b = idx_batch * DEV_BATCH_SIZE, (idx_batch + 1) * DEV_BATCH_SIZE
                    curr_batch = dev_set[dev_indices[s_b: e_b]]

                    indices = curr_batch["indices"]
                    curr_batch_device = {_k: _v.to(DEVICE) for _k, _v in curr_batch.items()
                                         if _k not in {"indices", "special_tokens_mask"}}
                    ground_truth = dev_set.targets(indices)
                    expected_types = ground_truth["met_type"]

                    probas = torch.softmax(model(**curr_batch_device)["logits"], dim=-1)
                    pred_logproba, correct_class = extract_pred_data(probas, expected_types,
                                                                     curr_batch["special_tokens_mask"].to(DEVICE))
                    pred_logproba = pred_logproba.cpu()
                    correct_class = correct_class.cpu()
                    dev_preds.append(torch.argmax(pred_logproba, dim=-1))
                    dev_true.append(correct_class)

            dev_preds = torch.cat(dev_preds).numpy()
            dev_true = torch.cat(dev_true).numpy()

            dev_metric = validation_fn(y_true=dev_true, y_pred=dev_preds)
            logging.info(f"\tValidation {args.validation_metric}: {dev_metric:.4f}")

            if dev_metric > best_dev_metric:
                logging.info(f"\t\tNew best, saving checkpoint!")
                model.save_pretrained(args.experiment_dir)
                best_dev_metric = dev_metric
                no_increase = 0
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
    model = AutoModelForTokenClassification.from_pretrained(args.experiment_dir).to(DEVICE)






