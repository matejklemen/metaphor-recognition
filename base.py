import logging
from time import time

import torch
import wandb
from torch import optim
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification

from data import TAG2ID, ID2TAG, extract_scheme_info
from utils import token_precision, token_recall, token_f1


class MetaphorController:
	def __init__(self, model_dir, label_scheme,
				 tokenizer_or_tokenizer_name="EMBEDDIA/sloberta", model_or_model_name="EMBEDDIA/sloberta",
				 learning_rate=2e-5, batch_size=32, validate_every_n_examples=3000, optimized_metric="f1_macro",
				 device="cuda"):
		self.model_dir = model_dir

		# Primary label scheme is the one used in model, secondary is the one used in the evaluation
		# They only differ if the primary one is IOB2-based (in that case, the secondary one is non-IOB2 equivalent)
		self.scheme_info = extract_scheme_info(label_scheme)
		self.prim_label_scheme = self.scheme_info["primary"]["name"]
		self.sec_label_scheme = self.scheme_info["secondary"]["name"]
		self.iob2 = self.scheme_info["iob2"]
		self.num_train_labels = 1 + self.scheme_info["primary"]["num_pos_labels"]
		self.num_eval_labels = self.scheme_info["secondary"]["num_pos_labels"]  # only eval on positive labels
		self.fallback_str = self.scheme_info["fallback_label"]

		self.validate_every_n_examples = validate_every_n_examples
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.dev_batch_size = self.batch_size * 2

		if device == "cuda" and not torch.cuda.is_available():
			raise ValueError(f"Device set to 'cuda', but CUDA-capable device was not found by torch")
		self.device = torch.device(device)
		self.device_str = device

		self.optimized_metric = optimized_metric

		if not isinstance(tokenizer_or_tokenizer_name, str):
			logging.info(f"Tokenizer was provided pre-loaded - it is assumed that the settings are pre-set.")
			self.tokenizer = tokenizer_or_tokenizer_name
		else:
			self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_or_tokenizer_name)
		self.tokenizer.save_pretrained(self.model_dir)

		if not isinstance(model_or_model_name, str):
			logging.info(f"Model was provided pre-loaded - it is assumed that the settings are pre-set.")
			self.model = model_or_model_name
		else:
			self.model = AutoModelForTokenClassification.from_pretrained(model_or_model_name,
																		 num_labels=self.num_train_labels)

		self.model = self.model.to(self.device)
		self.optimizer = optim.AdamW(params=self.model.parameters(), lr=self.learning_rate)

	def run_training(self, train_dataset, dev_dataset=None, num_epochs=1):
		ts = time()
		num_train, num_dev = len(train_dataset), 0
		dev_gt = {}
		if dev_dataset is not None:
			# TODO: extract boundaries of metaphors Set([(i_start, i_end), ...]) for entity-level evaluation?
			dev_gt = {"word_labels": dev_dataset.labels}
			num_dev = len(dev_dataset)
			if self.iob2:
				# Convert IOB2 to independent labels (remove B-, I-) for evaluation
				independent_labels = []
				for idx_dev in range(dev_dataset.labels.shape[0]):
					curr_labels = dev_dataset.labels[idx_dev]
					curr_processed = []
					for _lbl in curr_labels.tolist():
						if _lbl == -100:
							curr_processed.append(_lbl)
						else:
							str_tag = ID2TAG[self.prim_label_scheme][_lbl]
							if str_tag == self.fallback_str:
								curr_processed.append(TAG2ID[self.sec_label_scheme][str_tag])
							else:
								curr_processed.append(TAG2ID[self.sec_label_scheme][str_tag[2:]])  # without "B-" or "I-"

					independent_labels.append(curr_processed)

				independent_labels = torch.tensor(independent_labels)
				assert independent_labels.shape == dev_dataset.labels.shape
				dev_gt["word_labels"] = independent_labels

			dev_gt["word_labels"] = torch.tensor(dev_dataset.align_word_predictions(dev_dataset.labels, pad=True))

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
					continue

				dev_res = self.eval_pass(dev_dataset)
				dev_preds = torch.argmax(dev_res["pred_probas"], dim=-1).cpu()

				# If IOB2 is used, convert the labels to a non-IOB2 equivalent before token-level evaluation
				if self.iob2:
					# Convert IOB2 to independent labels (remove B-, I-) for evaluation
					independent_labels = []
					for idx_dev in range(dev_preds.shape[0]):
						curr_labels = dev_preds[idx_dev]
						curr_processed = []
						for _lbl in curr_labels.tolist():
							if _lbl == -100:
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
					assert independent_labels.shape == dev_preds.shape
					dev_preds = independent_labels

				dev_preds = torch.tensor(dev_dataset.align_word_predictions(dev_preds, pad=True))

				curr_dev_metrics = self.compute_metrics(true_labels=dev_gt["word_labels"],
														predicted_labels=dev_preds)
				curr_dev_metrics["loss"] = dev_res['loss'] / max(1, dev_res['num_batches'])
				logging.info(f"Dev loss: {curr_dev_metrics['loss']:.4f}")

				curr_dev_metrics_verbose = []
				for metric_name, metric_val in sorted(curr_dev_metrics.items(), key=lambda tup: tup[0]):
					curr_dev_metrics_verbose.append(f"{metric_name} = {metric_val:.4f}")
				curr_dev_metrics_verbose = "[Dev metrics] {}".format(", ".join(curr_dev_metrics_verbose))
				logging.info(curr_dev_metrics_verbose)
				wandb.log(curr_dev_metrics)

				curr_dev_metric = curr_dev_metrics[self.optimized_metric]
				if curr_dev_metric > best_dev_metric:
					logging.info(f"NEW BEST dev metric!")
					best_dev_metric = curr_dev_metric
					best_dev_metric_verbose = curr_dev_metrics_verbose
					self.model.save_pretrained(self.model_dir)

		logging.info(f"Training finished. Took {time() - ts:.4f}s")
		logging.info(f"Best validation metric: {best_dev_metric:.4f}")
		logging.info(best_dev_metric_verbose)
		wandb.summary[f"best_dev_{self.optimized_metric}"] = best_dev_metric

		# Reload best saved model
		self.model = AutoModelForTokenClassification.from_pretrained(self.model_dir).to(self.device)

	def compute_metrics(self, true_labels, predicted_labels):
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

	def run_prediction(self):
		pass

	def train_pass(self, train_data):
		self.model.train()
		train_loss, nb = 0.0, 0
		for curr_batch_cpu in tqdm(DataLoader(train_data, batch_size=self.batch_size)):
			curr_batch = {_k: _v.to(self.device) for _k, _v in curr_batch_cpu.items()}

			res = self.model(**curr_batch)
			loss = res["loss"]

			train_loss += float(loss)
			nb += 1

			loss.backward()
			self.optimizer.step()
			self.optimizer.zero_grad()

		return {"loss": train_loss, "num_batches": nb}

	@torch.no_grad()
	def eval_pass(self, dev_data, train_mode=False):
		if train_mode:
			self.model.train()
		else:
			self.model.eval()

		dev_loss, nb = 0.0, 0
		dev_probas = []
		for curr_batch_cpu in tqdm(DataLoader(dev_data, batch_size=self.dev_batch_size)):
			curr_batch = {_k: _v.to(self.device) for _k, _v in curr_batch_cpu.items()}

			res = self.model(**curr_batch)
			probas = torch.softmax(res["logits"], dim=-1)

			dev_loss += float(res["loss"])
			nb += 1

			dev_probas.append(probas)

		dev_probas = torch.cat(dev_probas)
		return {"pred_probas": dev_probas, "loss": dev_loss, "num_batches": nb}
