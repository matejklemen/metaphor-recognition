import argparse
import json
import logging
import os
import sys
import time
from copy import deepcopy

import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import trange
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from data import load_df, TransformersSentenceDataset
from utils import visualize_sentence_predictions

MAX_THRESH_TO_CHECK = 100

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_dir", type=str, default="debug_sent_modeling")
parser.add_argument("--data_path", type=str, default="data/komet-sent-traindev/dev.tsv")

parser.add_argument("--pretrained_name_or_path", type=str, default="EMBEDDIA/sloberta")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--decision_threshold_bin", type=float, default=None,
                    help="Specify a decision threshold to be used in binary classification of metaphors")
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
        "mrwi": True, "mrwd": True, "mflag": False, "widli": False, "mrwimp": True, "bridge": False,
        "history_prev_sents": 0, "max_length": 64
    }

    if os.path.exists(args.pretrained_name_or_path):
        logging.info("Loading existing config information from experiment_config.json")
        with open(os.path.join(args.pretrained_name_or_path, "experiment_config.json"), "r") as f:
            existing_config = json.load(f)

        # Override some arguments
        for override_key in ["experiment_dir", "data_path", "pretrained_name_or_path", "use_cpu", "random_seed",
                             "batch_size", "decision_threshold_bin"]:
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

    # TODO: maybe this encoding should be saved in the model directory as well? And this should just be a default option
    MET_TYPE_MAPPING = {_type: "metaphor" for _type in valid_types}  # everything else maps to "O"
    type_encoding = {"O": 0, "metaphor": 1}
    rev_type_encoding = {_idx: _type for _type, _idx in type_encoding.items()}
    num_types = len(type_encoding)

    frame_encoding = None  # TODO: implement me

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.pretrained_name_or_path).to(DEVICE)

    df_test = load_df(args.data_path)

    has_type = "met_type" in df_test.columns
    if has_type:
        # Transform the metaphor annotations according to the scheme and drop uninteresting metaphor types
        for idx_ex in range(df_test.shape[0]):
            curr_ex = df_test.iloc[idx_ex]

            new_met_type = []
            for met_info in curr_ex["met_type"]:
                new_type = MET_TYPE_MAPPING.get(met_info["type"], "O")
                if new_type != "O":
                    met_info_copy = deepcopy(met_info)
                    met_info_copy["type"] = new_type
                    new_met_type.append(met_info_copy)

            df_test.at[idx_ex, "met_type"] = new_met_type

    test_set = TransformersSentenceDataset.from_dataframe(df_test, history_prev_sents=args.history_prev_sents,
                                                          type_encoding=type_encoding, frame_encoding=frame_encoding,
                                                          max_length=args.max_length,
                                                          tokenizer_or_tokenizer_name=tokenizer)
    logging.info(f"Loaded {len(test_set)} test instances")
    if has_type:
        test_set.target_data["met_type"] = (test_set.target_data["met_type"][:, 1] > 0).long()

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

                curr_batch_device = {_k: _v.to(DEVICE) for _k, _v in curr_batch.items()
                                     if _k not in {"indices", "special_tokens_mask"}}

                probas = torch.softmax(model(**curr_batch_device)["logits"], dim=-1)
                round_probas.append(probas.cpu())

            test_probas.append(torch.cat(round_probas))

        test_probas = torch.stack(test_probas)  # [num_rounds, len(test_set) , num_types]
        mean_probas = torch.mean(test_probas, dim=0)
        if num_rounds < 2:
            sd_probas = torch.zeros_like(mean_probas)
        else:
            sd_probas = torch.std(test_probas, dim=0)

        df_test["mean_proba"] = mean_probas[test_set.alignment_indices].tolist()
        df_test["sd_proba"] = sd_probas[test_set.alignment_indices].tolist()

    if args.decision_threshold_bin is not None:
        logging.info(f"Using manually specified decision threshold T={args.decision_threshold_bin}")
        test_preds = (mean_probas[:, 1] >= args.decision_threshold_bin).int()
    else:
        logging.warning("Decision threshold was neither automatically nor manually determined, using argmax")
        test_preds = torch.argmax(mean_probas, dim=-1)

    if has_type:
        # Precision-recall curve
        import matplotlib.pyplot as plt
        from sklearn.metrics import PrecisionRecallDisplay
        PrecisionRecallDisplay.from_predictions(y_true=test_set.target_data["met_type"].numpy(),
                                                y_pred=mean_probas[:, 1].numpy(),
                                                pos_label=1,
                                                name=args.pretrained_name_or_path)
        plt.savefig(os.path.join(args.experiment_dir, "pr_curve.png"))
        plt.clf()

        # Threshold-F1 curve: useful when evaluating model on a validation set
        thresh_to_check = sorted(list(set(mean_probas[:, 1].tolist())))
        thresh_to_check = np.percentile(thresh_to_check, q=list(range(MAX_THRESH_TO_CHECK + 1)),
                                        method="closest_observation").tolist()

        # Holds (<thresh>, P, R, F1) for each threshold
        thresh_stats = []
        for curr_thresh in thresh_to_check:
            np_test_preds = (mean_probas[:, 1] >= curr_thresh).int().numpy()
            np_test_true = test_set.target_data["met_type"].numpy()

            thresh_stats.append((curr_thresh,
                                 precision_score(y_true=np_test_true, y_pred=np_test_preds),
                                 recall_score(y_true=np_test_true, y_pred=np_test_preds),
                                 f1_score(y_true=np_test_true, y_pred=np_test_preds)))

        thresh_stats = sorted(thresh_stats, key=lambda curr_stats: curr_stats[0])
        plt.title("F1 score at different thresholds")
        plt.plot(thresh_stats, list(map(lambda curr_stats: curr_stats[3], thresh_stats)), color="blue",
                 linestyle="--")
        plt.xlim([0.0, 1.0 + 0.01])
        plt.ylim([0.0, 1.0 + 0.01])
        plt.xlabel("Threshold")
        plt.ylabel("F1")
        plt.savefig(os.path.join(args.experiment_dir, "f1_curve.png"))

        logging.info("**********************")
        prec_sorted_stats = sorted(thresh_stats, key=lambda curr_stats: (curr_stats[1], curr_stats[2]))
        _T, _P, _R, _F1 = prec_sorted_stats[-1]
        logging.info(f"[Maximize precision] T = {_T}, validation P = {_P:.4f}, R = {_R:.4f}, F1 = {_F1:.4f}")

        rec_sorted_stats = sorted(thresh_stats, key=lambda curr_stats: (curr_stats[2], curr_stats[1]))
        _T, _P, _R, _F1 = rec_sorted_stats[-1]
        logging.info(f"[Maximize recall] T = {_T}, validation P = {_P:.4f}, R = {_R:.4f}, F1 = {_F1:.4f}")

        f1_sorted_stats = sorted(thresh_stats, key=lambda curr_stats: (curr_stats[3], curr_stats[0]))
        _T, _P, _R, _F1 = f1_sorted_stats[-1]
        logging.info(f"[Maximize F1] T = {_T}, validation P = {_P:.4f}, R = {_R:.4f}, F1 = {_F1:.4f}")
        logging.info("IMPORTANT: This maximization is not automatically taken into account in the final test metrics")
        logging.info("**********************")

        final_test_p = precision_score(y_true=test_set.target_data["met_type"].numpy(),
                                       y_pred=test_preds.numpy())
        final_test_r = recall_score(y_true=test_set.target_data["met_type"].numpy(),
                                    y_pred=test_preds.numpy())
        final_test_f1 = f1_score(y_true=test_set.target_data["met_type"].numpy(),
                                 y_pred=test_preds.numpy())
        logging.info(f"Test metrics using model: "
                     f"P = {final_test_p:.4f}, R = {final_test_r:.4f}, F1 = {final_test_f1:.4f}")
    else:
        logging.info("Skipping test set evaluation because no 'met_type' column is provided in the test file")

    with open(os.path.join(args.experiment_dir, "pred_visualization_test.html"), "w", encoding="utf-8") as f:
        test_sents = list(map(lambda _sent: " ".join(_sent),
                              test_set.input_sentences))
        test_preds_str = list(map(lambda _curr_pred: rev_type_encoding[_curr_pred],
                                  test_preds.tolist()))
        if has_type:
            test_true_str = list(map(lambda _curr_pred: rev_type_encoding[_curr_pred],
                                     test_set.target_data["met_type"].tolist()))
            test_true_str = [test_true_str[_i] for _i in test_set.alignment_indices]
        else:
            test_true_str = None

        # Align encoded examples and their predictions to the order in the source dataframe
        test_sents = [test_sents[_i] for _i in test_set.alignment_indices]
        test_preds_str = [test_preds_str[_i] for _i in test_set.alignment_indices]

        visualization_html = visualize_sentence_predictions(test_sents,
                                                            labels_predicted=test_preds_str,
                                                            labels_true=test_true_str)
        print(visualization_html, file=f)

    df_test["preds_transformed"] = test_preds_str
    if has_type:
        df_test["true_transformed"] = test_true_str

    df_test.to_csv(os.path.join(args.experiment_dir, "test_results.tsv"), index=False, sep="\t")













