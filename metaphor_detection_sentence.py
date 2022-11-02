import argparse
import itertools
import json
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import trange
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import wandb
from data import TransformersSentenceDataset, load_df, create_examples
from utils import visualize_sentence_predictions

MAX_THRESH_TO_CHECK = 100


parser = argparse.ArgumentParser()
parser.add_argument("--experiment_dir", type=str, default="debug_sent_modeling")
parser.add_argument("--mode", type=str, choices=["train", "eval"], default="train")
parser.add_argument("--train_path", type=str,
                    default="/home/matej/Documents/metaphor-detection/data/komet/train.tsv")
parser.add_argument("--dev_path", type=str,
                    default="/home/matej/Documents/metaphor-detection/data/komet/dev.tsv")
parser.add_argument("--test_path", type=str,
                    default="/home/matej/Documents/metaphor-detection/data/komet/test.tsv")
parser.add_argument("--history_prev_sents", type=int, default=0)
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
parser.add_argument("--decision_threshold_bin", type=float, default=None,
                    help="Specify a decision threshold to be used in binary classification of metaphors")
parser.add_argument("--tune_last_only", action="store_true")

parser.add_argument("--wandb_project_name", type=str, default="metaphor-komet-sentence")
parser.add_argument("--random_seed", type=int, default=17)
parser.add_argument("--use_cpu", action="store_true")

if __name__ == "__main__":
    args = parser.parse_args()
    if os.path.exists(args.experiment_dir) and args.mode == "train":
        raise ValueError("--mode=train, but the experiment_dir exists, so the data could accidentally get overriden."
                         "Please remove the experiment_dir manually and rerun the script")

    os.makedirs(args.experiment_dir, exist_ok=True)

    # Set up logging to file and stdout
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for curr_handler in [logging.StreamHandler(sys.stdout),
                         logging.FileHandler(os.path.join(args.experiment_dir, f"{args.mode}.log"))]:
        curr_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s"))
        logger.addHandler(curr_handler)

    if args.mode == "eval":
        with open(os.path.join(args.experiment_dir, "experiment_config.json"), "r") as f:
            existing_config = json.load(f)
        logging.info("Loading existing config information from experiment_config.json")

        existing_config.pop("test_path", None)
        existing_config.pop("pretrained_name_or_path", None)
        existing_config.pop("batch_size", None)

        if args.decision_threshold_bin is not None:
            existing_config.pop("decision_threshold_bin", None)

        for k, v in existing_config.items():
            setattr(args, k, v)
        args.mode = "eval"

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

    MET_TYPE_MAPPING = {_type: "metaphor" for _type in valid_types}  # everything else maps to "O"
    type_encoding = {"O": 0, "metaphor": 1}
    rev_type_encoding = {_idx: _type for _type, _idx in type_encoding.items()}
    num_types = len(type_encoding)

    frame_encoding = None  # TODO: implement me

    wandb.init(project=args.wandb_project_name, config=vars(args))

    best_thresh = None
    if args.mode == "train":
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_name_or_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.pretrained_name_or_path, num_labels=num_types
        ).to(DEVICE)

        if args.tune_last_only:
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
        _train_instances = create_examples(df_train, history_prev_sents=args.history_prev_sents)
        train_lengths = sorted([len(_curr)
                                for _curr in
                                tokenizer.batch_encode_plus(list(map(lambda _inst: _inst.words, _train_instances)),
                                                            is_split_into_words=True)["input_ids"]])
        max_length = train_lengths[int(0.99 * len(train_lengths))]
        args.max_length = max_length

        train_set = TransformersSentenceDataset.from_dataframe(df_train, history_prev_sents=args.history_prev_sents,
                                                               type_encoding=type_encoding, frame_encoding=frame_encoding,
                                                               max_length=args.max_length,
                                                               tokenizer_or_tokenizer_name=tokenizer)
        dev_set = TransformersSentenceDataset.from_dataframe(df_dev, history_prev_sents=args.history_prev_sents,
                                                             type_encoding=type_encoding, frame_encoding=frame_encoding,
                                                             max_length=args.max_length,
                                                             tokenizer_or_tokenizer_name=tokenizer)

        # If there's at least one metaphor annotated in the sentence, mark example as metaphorical
        train_set.target_data["met_type"] = (train_set.target_data["met_type"][:, 1] > 0).long()
        dev_set.target_data["met_type"] = (dev_set.target_data["met_type"][:, 1] > 0).long()

        logging.info(f"Loaded {len(train_set)} training instances, {len(dev_set)} validation instances")

        loss_fn = nn.CrossEntropyLoss()
        validation_fn = lambda y_true, y_pred: 0.0  # placeholder
        if args.validation_metric == "f1_score_binary":
            validation_fn = lambda y_true, y_pred: f1_score(y_true=y_true, y_pred=y_pred, pos_label=1)
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
                    s_b, e_b = idx_batch * args.batch_size,  (idx_batch + 1) * args.batch_size
                    curr_batch = train_set[sub_indices[s_b: e_b]]

                    indices = curr_batch["indices"]
                    curr_batch_device = {_k: _v.to(DEVICE) for _k, _v in curr_batch.items()
                                         if _k not in {"indices", "special_tokens_mask"}}
                    ground_truth = train_set.targets(indices)

                    logits = model(**curr_batch_device)["logits"]

                    loss = loss_fn(logits, ground_truth["met_type"].to(DEVICE))
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

                    num_dev_batches = (len(dev_set) + DEV_BATCH_SIZE - 1) // DEV_BATCH_SIZE
                    dev_indices = torch.arange(len(dev_set))

                    dev_preds = []

                    for idx_batch in trange(num_dev_batches):
                        s_b, e_b = idx_batch * DEV_BATCH_SIZE, (idx_batch + 1) * DEV_BATCH_SIZE
                        batch_indices = dev_indices[s_b: e_b]
                        curr_batch = dev_set[batch_indices]

                        indices = curr_batch["indices"]
                        curr_batch_device = {_k: _v.to(DEVICE) for _k, _v in curr_batch.items()
                                             if _k not in {"indices", "special_tokens_mask"}}

                        probas = torch.softmax(model(**curr_batch_device)["logits"], dim=-1)
                        dev_preds.append(torch.argmax(probas, dim=-1).cpu())

                dev_preds = torch.cat(dev_preds)

                dev_metric = validation_fn(y_true=dev_set.target_data["met_type"].numpy(), y_pred=dev_preds.numpy())
                logging.info(f"\tValidation {args.validation_metric}: {dev_metric:.4f}")
                wandb.log({f"dev_{args.validation_metric}": dev_metric})

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

        wandb.summary[f"best_dev_{args.validation_metric}"] = best_dev_metric
        logging.info(f"Best validation {args.validation_metric}: {best_dev_metric:.4f}")

        logging.info("Reloading best model for evaluation...")
        model = AutoModelForSequenceClassification.from_pretrained(args.experiment_dir).to(DEVICE)

        # Obtain predictions on validation set with the best model, save them, visualize them
        with torch.inference_mode():
            model.eval()

            num_dev_batches = (len(dev_set) + DEV_BATCH_SIZE - 1) // DEV_BATCH_SIZE
            dev_indices = torch.arange(len(dev_set))
            dev_preds, dev_probas = [], []

            for idx_batch in trange(num_dev_batches):
                s_b, e_b = idx_batch * DEV_BATCH_SIZE, (idx_batch + 1) * DEV_BATCH_SIZE
                batch_indices = dev_indices[s_b: e_b]
                curr_batch = dev_set[batch_indices]

                indices = curr_batch["indices"]
                curr_batch_device = {_k: _v.to(DEVICE) for _k, _v in curr_batch.items()
                                     if _k not in {"indices", "special_tokens_mask"}}

                probas = torch.softmax(model(**curr_batch_device)["logits"], dim=-1)

                dev_preds.append(torch.argmax(probas, dim=-1).cpu())
                dev_probas.append(probas.cpu())

        dev_probas = torch.cat(dev_probas)
        if args.optimize_bin_threshold:
            thresh_to_check = sorted(list(set(dev_probas[:, 1].tolist())))
            if len(thresh_to_check) > MAX_THRESH_TO_CHECK:
                step_size = len(thresh_to_check) // MAX_THRESH_TO_CHECK
                thresh_to_check = thresh_to_check[::step_size][:MAX_THRESH_TO_CHECK]

            best_thresh, metric_with_best_thresh = None, 0.0
            for curr_thresh in thresh_to_check:
                dev_preds = (dev_probas[:, 1] >= curr_thresh).int()
                curr_metric = validation_fn(y_true=dev_set.target_data["met_type"].numpy(), y_pred=dev_preds.numpy())
                if curr_metric > metric_with_best_thresh:
                    best_thresh = curr_thresh
                    metric_with_best_thresh = curr_metric

            wandb.log({f"best_dev_{args.validation_metric}": metric_with_best_thresh})
            logging.info(f"[Threshold optimization] Best T={best_thresh}, validation {args.validation_metric} = {metric_with_best_thresh:.4f}")
            dev_preds = (dev_probas[:, 1] >= best_thresh).int()
        else:
            dev_preds = torch.argmax(dev_probas, dim=-1)

        final_dev_p = precision_score(y_true=dev_set.target_data["met_type"].numpy(),
                                      y_pred=dev_preds.numpy())
        final_dev_r = recall_score(y_true=dev_set.target_data["met_type"].numpy(),
                                   y_pred=dev_preds.numpy())
        final_dev_f1 = f1_score(y_true=dev_set.target_data["met_type"].numpy(),
                                y_pred=dev_preds.numpy())

        logging.info(f"Dev metrics using best model: P = {final_dev_p:.4f}, R = {final_dev_r:.4f}, F1 = {final_dev_f1:.4f}")

        with open(os.path.join(args.experiment_dir, "pred_visualization.html"), "w", encoding="utf-8") as f:
            dev_sents = list(map(lambda _sent: " ".join(_sent), dev_set.input_sentences))
            dev_preds_str = list(map(lambda _curr_pred: rev_type_encoding[_curr_pred],
                                     dev_preds.tolist()))
            dev_true_str = list(map(lambda _curr_pred: rev_type_encoding[_curr_pred],
                                    dev_set.target_data["met_type"].tolist()))

            visualization_html = visualize_sentence_predictions(dev_sents,
                                                                labels_predicted=dev_preds_str,
                                                                labels_true=dev_true_str)
            print(visualization_html, file=f)

        df_dev["preds_transformed"] = dev_preds_str
        df_dev["true_transformed"] = dev_true_str
        df_dev.to_csv(os.path.join(args.experiment_dir, "dev_results.tsv"), index=False, sep="\t")

    if args.mode == "eval":
        tokenizer = AutoTokenizer.from_pretrained(args.experiment_dir)
        model = AutoModelForSequenceClassification.from_pretrained(args.experiment_dir).to(DEVICE)

    df_test = load_df(args.test_path)
    has_type = "met_type" in df_test.columns

    if has_type:
        for idx_ex in range(df_test.shape[0]):
            new_met_type = []
            for met_info in df_test.iloc[idx_ex]["met_type"]:
                new_type = MET_TYPE_MAPPING.get(met_info["type"], "O")
                if new_type != "O":
                    met_info["type"] = new_type
                    new_met_type.append(met_info)

            df_test.at[idx_ex, "met_type"] = new_met_type

    test_set = TransformersSentenceDataset.from_dataframe(df_test, history_prev_sents=args.history_prev_sents,
                                                          type_encoding=type_encoding, frame_encoding=frame_encoding,
                                                          max_length=args.max_length,
                                                          tokenizer_or_tokenizer_name=tokenizer)
    if has_type:
        test_set.target_data["met_type"] = (test_set.target_data["met_type"][:, 1] > 0).long()

    logging.info(f"Loaded {len(test_set)} test instances")

    # Obtain predictions on TEST set with the best model, save them, visualize them
    with torch.inference_mode():
        model.eval()

        num_test_batches = (len(test_set) + DEV_BATCH_SIZE - 1) // DEV_BATCH_SIZE
        test_indices = torch.arange(len(test_set))
        test_probas = []

        for idx_batch in trange(num_test_batches):
            s_b, e_b = idx_batch * DEV_BATCH_SIZE, (idx_batch + 1) * DEV_BATCH_SIZE
            batch_indices = test_indices[s_b: e_b]
            curr_batch = test_set[batch_indices]

            indices = curr_batch["indices"]
            curr_batch_device = {_k: _v.to(DEVICE) for _k, _v in curr_batch.items()
                                 if _k not in {"indices", "special_tokens_mask"}}

            probas = torch.softmax(model(**curr_batch_device)["logits"], dim=-1)
            test_probas.append(probas.cpu())

    test_probas = torch.cat(test_probas)
    # If only running the evaluation part, allow the user to specify a custom decision threshold
    if best_thresh is None and args.decision_threshold_bin is not None:
        best_thresh = args.decision_threshold_bin
        logging.info(f"Using manually specified decision threshold T={best_thresh}")

    # If the decision threshold is neither automatically or manually determined, default to argmax
    if best_thresh is None:
        logging.warning("Decision threshold was neither automatically nor manually determined, using argmax")
        test_preds = torch.argmax(test_probas, dim=-1)
    else:
        test_preds = (test_probas[:, 1] >= best_thresh).int()

    if has_type:
        final_test_p = precision_score(y_true=test_set.target_data["met_type"].numpy(),
                                       y_pred=test_preds.numpy())
        final_test_r = recall_score(y_true=test_set.target_data["met_type"].numpy(),
                                    y_pred=test_preds.numpy())
        final_test_f1 = f1_score(y_true=test_set.target_data["met_type"].numpy(),
                                 y_pred=test_preds.numpy())
        logging.info(f"Test metrics using best model: P = {final_test_p:.4f}, R = {final_test_r:.4f}, F1 = {final_test_f1:.4f}")
        wandb.log({
            "test_p": final_test_p,
            "test_r": final_test_r,
            "test_f1": final_test_f1
        })
    else:
        logging.info("Skipping test set evaluation because no 'met_type' column is provided in the test file")

    with open(os.path.join(args.experiment_dir, "pred_visualization_test.html"), "w", encoding="utf-8") as f:
        test_sents = list(map(lambda _sent: " ".join(_sent), test_set.input_sentences))
        test_preds_str = list(map(lambda _curr_pred: rev_type_encoding[_curr_pred],
                                  test_preds.tolist()))
        if has_type:
            test_true_str = list(map(lambda _curr_pred: rev_type_encoding[_curr_pred],
                                     test_set.target_data["met_type"].tolist()))
        else:
            test_true_str = None

        visualization_html = visualize_sentence_predictions(test_sents,
                                                            labels_predicted=test_preds_str,
                                                            labels_true=test_true_str)
        print(visualization_html, file=f)

    df_test["preds_transformed"] = test_preds_str
    if has_type:
        df_test["true_transformed"] = test_true_str

    df_test.to_csv(os.path.join(args.experiment_dir, "test_results.tsv"), index=False, sep="\t")

    if args.mode == "train":
        with open(os.path.join(args.experiment_dir, "experiment_config.json"), "w") as f:
            json.dump(vars(args), fp=f, indent=4)

    wandb.finish()


