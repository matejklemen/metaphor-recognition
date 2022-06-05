import json
import logging
import os
from time import time

import torch
import wandb
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm

from base import MetaphorController
from custom_modules import AutoModelForTokenMultiClassification
from data import extract_scheme_info, ID2TAG
from utils import token_precision, token_recall, token_f1


class MetaphorMultiTaskController(MetaphorController):
	def __init__(self, model_dir, label_scheme,
				 tokenizer_or_tokenizer_name="EMBEDDIA/sloberta", model_or_model_name="EMBEDDIA/sloberta",
				 learning_rate=2e-5, batch_size=32, validate_every_n_examples=3000, optimized_metric="f1_macro",
				 device="cuda", **kwargs):
		self.scheme_info = extract_scheme_info(label_scheme)
		if self.scheme_info["iob2"]:
			raise NotImplementedError(f"Frame encoding is not adapted for IOB2")

		self.frame2id = kwargs["frame_encoding"]
		self.id2frame = {_i: _frame for _frame, _i in self.frame2id.items()}
		self.num_train_labels = 1 + self.scheme_info["primary"]["num_pos_labels"]
		self.num_train_frames = len(self.frame2id)
		self.num_eval_frames = self.num_train_frames - 1  # all but "O"

		if not isinstance(model_or_model_name, str):
			logging.info(f"Model was provided pre-loaded - it is assumed that the settings are pre-set.")
			used_model = model_or_model_name
		else:
			# num_types: int, num_frames: int
			used_model = AutoModelForTokenMultiClassification(model_or_model_name,
															  num_types=self.num_train_labels,
															  num_frames=self.num_train_frames)

		super().__init__(
			model_dir=model_dir, label_scheme=label_scheme, tokenizer_or_tokenizer_name=tokenizer_or_tokenizer_name,
			model_or_model_name=used_model, learning_rate=learning_rate, batch_size=batch_size,
			validate_every_n_examples=validate_every_n_examples, optimized_metric=optimized_metric, device=device
		)

	def save(self):
		with open(os.path.join(self.model_dir, "controller_config.json"), "w", encoding="utf-8") as f_config:
			json.dump({
				"label_scheme": self.label_scheme,
				"frame_encoding": self.frame2id,
				"learning_rate": self.learning_rate,
				"batch_size": self.batch_size,
				"validate_every_n_examples": self.validate_every_n_examples,
				"optimized_metric": self.optimized_metric,
				"device": self.device_str
			}, fp=f_config, indent=4)

		self.model.save_pretrained(self.model_dir)

	@staticmethod
	def load(controller_dir, **override_kwargs):
		with open(os.path.join(controller_dir, "controller_config.json"), "r", encoding="utf-8") as f_config:
			config = json.load(fp=f_config)

		config["model_dir"] = controller_dir
		# When training, these likely point to a HF checkpoint, but they get saved locally
		config["model_or_model_name"] = controller_dir
		config["tokenizer_or_tokenizer_name"] = controller_dir

		# User may override pre-trained config parameters, such as batch size or device
		for config_key, new_value in override_kwargs.items():
			old_value = ""
			if config_key in config:
				old_value = f" {config_key} = {config[config_key]} -> "
			logging.info(f"Overriding config:{old_value}{config_key} = {new_value}")
			config[config_key] = new_value

		return MetaphorMultiTaskController(**config)

	def run_training(self, train_dataset, dev_dataset=None, num_epochs=1):
		ts = time()
		num_train, num_dev = len(train_dataset), 0
		dev_gt = {}
		if dev_dataset is not None:
			num_dev = len(dev_dataset)

			dev_gt["word_labels"] = torch.tensor(dev_dataset.align_word_predictions(dev_dataset.labels, pad=True))
			dev_gt["word_frame_labels"] = torch.tensor(dev_dataset.align_word_predictions(dev_dataset.frame_labels, pad=True))

		# Save initial version to prevent crash in case a non-trivial model can not be trained
		self.save()

		best_dev_metric, best_dev_metric_verbose = 0.0, None
		max_subset_size = self.validate_every_n_examples
		num_train_subsets = (num_train + max_subset_size - 1) // max_subset_size
		for idx_epoch in range(num_epochs):
			logging.info(f"Epoch #{1 + idx_epoch}/{num_epochs}")
			train_loss, nb = 0.0, 0

			rand_indices = torch.randperm(num_train)
			for idx_sub in range(num_train_subsets):
				logging.info(f"Subset #{1 + idx_sub}/{num_train_subsets}")
				s_sub, e_sub = idx_sub * max_subset_size, (idx_sub + 1) * max_subset_size
				curr_sub = Subset(train_dataset, rand_indices[s_sub: e_sub])

				train_res = self.train_pass(curr_sub)
				train_loss += train_res["loss"]
				nb += train_res["num_batches"]

				logging.info(f"[Train] loss: {train_loss / max(1, nb):.4f}")

				# Skip evaluation on dev set if no dev set is provided or if training on small leftover training subset
				if num_dev == 0 or len(curr_sub) < max_subset_size // 2:
					# If no validation set is provided, new model is always saved
					if num_dev == 0:
						self.save()
					continue

				dev_res = self.eval_pass(dev_dataset, batch_size=(2 * self.batch_size))
				dev_type_preds = torch.argmax(dev_res["pred_probas_type"], dim=-1).cpu()
				dev_type_preds = torch.tensor(dev_dataset.align_word_predictions(dev_type_preds, pad=True))

				dev_frame_preds = torch.argmax(dev_res["pred_probas_frame"], dim=-1).cpu()
				dev_frame_preds = torch.tensor(dev_dataset.align_word_predictions(dev_frame_preds, pad=True))

				curr_dev_metrics = {
					"metaphor_type": self.compute_type_metrics(true_labels=dev_gt["word_labels"], predicted_labels=dev_type_preds),
					"metaphor_frame": self.compute_frame_metrics(true_labels=dev_gt["word_frame_labels"], predicted_labels=dev_frame_preds)
				}
				wandb_dev_metrics = {}
				# For compatibility with single-task learning logging, store metaphor_type metrics as primary metrics
				for metric_name, metric_val in sorted(curr_dev_metrics["metaphor_type"].items(), key=lambda tup: tup[0]):
					wandb_dev_metrics[metric_name] = metric_val

				for metric_name, metric_val in sorted(curr_dev_metrics["metaphor_frame"].items(), key=lambda tup: tup[0]):
					wandb_dev_metrics[f"metaphor_frame_{metric_name}"] = metric_val

				curr_dev_metrics["loss"] = dev_res['loss'] / max(1, dev_res['num_batches'])
				wandb_dev_metrics["loss"] = curr_dev_metrics["loss"]
				logging.info(f"Dev loss: {curr_dev_metrics['loss']:.4f}")

				curr_dev_metrics_verbose = []
				for metric_info in sorted(curr_dev_metrics.items(), key=lambda tup: tup[0]):
					curr_dev_metrics_verbose.append(str(metric_info))
				curr_dev_metrics_verbose = f"[Dev metrics] {curr_dev_metrics}"
				logging.info(curr_dev_metrics_verbose)
				wandb.log(wandb_dev_metrics)

				# NOTE: we (currently) don't optimize frame metrics, it is just used as "regularization"
				curr_dev_metric = curr_dev_metrics["metaphor_type"][self.optimized_metric]
				if curr_dev_metric > best_dev_metric:
					logging.info(f"NEW BEST dev metric!")
					best_dev_metric = curr_dev_metric
					best_dev_metric_verbose = curr_dev_metrics_verbose
					self.save()

		logging.info(f"Training finished. Took {time() - ts:.4f}s")
		logging.info(f"Best validation metric: {best_dev_metric:.4f}")
		logging.info(best_dev_metric_verbose)
		wandb.summary[f"best_dev_{self.optimized_metric}"] = best_dev_metric

		# Reload best saved model
		self.model = AutoModelForTokenMultiClassification(self.model_dir,
														  num_types=self.num_train_labels,
														  num_frames=self.num_train_frames).to(self.device)

	# TODO: quick hack, redundant code for computing metrics
	def compute_type_metrics(self, true_labels, predicted_labels):
		metrics = {}
		pos_labels = list(range(1, 1 + self.num_eval_labels))  # all but the fallback label (eval on pos labels)
		macro_p, macro_r, macro_f1 = 0.0, 0.0, 0.0
		for curr_label in pos_labels:
			curr_label_str = ID2TAG[self.sec_label_scheme][curr_label]
			metrics[f"p_{curr_label_str}"] = token_precision(true_labels, predicted_labels, pos_label=curr_label)
			metrics[f"r_{curr_label_str}"] = token_recall(true_labels, predicted_labels, pos_label=curr_label)
			metrics[f"f1_{curr_label_str}"] = token_f1(true_labels, predicted_labels, pos_label=curr_label)

			macro_p += metrics[f"p_{curr_label_str}"]
			macro_r += metrics[f"r_{curr_label_str}"]
			macro_f1 += metrics[f"f1_{curr_label_str}"]

		metrics[f"p_macro"] = macro_p / max(1, len(pos_labels))
		metrics[f"r_macro"] = macro_r / max(1, len(pos_labels))
		metrics[f"f1_macro"] = macro_f1 / max(1, len(pos_labels))
		return metrics

	def compute_frame_metrics(self, true_labels, predicted_labels):
		metrics = {}
		pos_labels = list(range(1, 1 + self.num_eval_frames))  # all but the fallback label (eval on pos labels)
		macro_p, macro_r, macro_f1 = 0.0, 0.0, 0.0
		for curr_label in pos_labels:
			curr_label_str = self.id2frame[curr_label]
			metrics[f"p_{curr_label_str}"] = token_precision(true_labels, predicted_labels, pos_label=curr_label)
			metrics[f"r_{curr_label_str}"] = token_recall(true_labels, predicted_labels, pos_label=curr_label)
			metrics[f"f1_{curr_label_str}"] = token_f1(true_labels, predicted_labels, pos_label=curr_label)

			macro_p += metrics[f"p_{curr_label_str}"]
			macro_r += metrics[f"r_{curr_label_str}"]
			macro_f1 += metrics[f"f1_{curr_label_str}"]

		metrics[f"p_macro"] = macro_p / max(1, len(pos_labels))
		metrics[f"r_macro"] = macro_r / max(1, len(pos_labels))
		metrics[f"f1_macro"] = macro_f1 / max(1, len(pos_labels))
		return metrics

	def run_prediction(self, test_data, mcd_iters=0):
		raise NotImplementedError()
		assert mcd_iters >= 0
		use_mcd = mcd_iters > 0
		eff_iters = 1 if mcd_iters == 0 else mcd_iters

		pred_probas = []
		for idx_iter in range(eff_iters):
			curr_res = self.eval_pass(test_data, train_mode=use_mcd)
			pred_probas.append(curr_res["pred_probas"])

		pred_probas = torch.stack(pred_probas)  # [eff_iters, len(test_data), max_length, num_train_labels]
		mean_probas = torch.mean(pred_probas, dim=0)  # [len(test_data), max_length, num_train_labels]
		preds = torch.argmax(mean_probas, dim=-1)  # [len(test_data), max_length]

		# If IOB2 is used, convert the labels to a non-IOB2 equivalent
		if self.iob2:
			# Convert IOB2 to independent labels (remove B-, I-) for evaluation
			independent_labels = []
			for idx_ex in range(preds.shape[0]):
				curr_labels = preds[idx_ex]
				curr_processed = []
				for _lbl in curr_labels.tolist():
					if _lbl == LOSS_IGNORE_INDEX:
						curr_processed.append(_lbl)
					else:
						str_tag = ID2TAG[self.prim_label_scheme][_lbl]
						if str_tag == self.fallback_str:
							curr_processed.append(TAG2ID[self.sec_label_scheme][str_tag])
						else:
							# without "B-" or "I-"
							curr_processed.append(TAG2ID[self.sec_label_scheme][str_tag[2:]])

				independent_labels.append(curr_processed)

			independent_labels = torch.tensor(independent_labels)
			assert independent_labels.shape == preds.shape
			preds = independent_labels

		return {
			"pred_probas": pred_probas,
			"preds": preds
		}

	@torch.no_grad()
	def eval_pass(self, dev_data, train_mode=False, batch_size=None):
		used_batch_size = self.batch_size if batch_size is None else int(batch_size)
		if train_mode:
			self.model.train()
		else:
			self.model.eval()

		dev_loss, nb = 0.0, 0
		type_probas = []
		frame_probas = []
		for curr_batch_cpu in tqdm(DataLoader(dev_data, batch_size=used_batch_size)):
			curr_batch = {_k: _v.to(self.device) for _k, _v in curr_batch_cpu.items()}

			res = self.model(**curr_batch)

			dev_loss += float(res["loss"])
			nb += 1

			type_probas.append(torch.softmax(res["logits"]["metaphor_type"], dim=-1))
			frame_probas.append(torch.softmax(res["logits"]["metaphor_frame"], dim=-1))

		type_probas = torch.cat(type_probas)
		frame_probas = torch.cat(frame_probas)
		return {"pred_probas_type": type_probas, "pred_probas_frame": frame_probas, "loss": dev_loss, "num_batches": nb}
