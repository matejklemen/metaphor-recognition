from typing import List

from sklearn.metrics import precision_score, recall_score, f1_score


def token_precision(true_labels, pred_labels, pos_label: int = 1, ignore_label: int = -100):
	valid_mask = true_labels != ignore_label

	valid_true = true_labels[valid_mask] == pos_label
	valid_pred = pred_labels[valid_mask] == pos_label
	return precision_score(y_true=valid_true, y_pred=valid_pred, average="binary", pos_label=1)


def token_recall(true_labels, pred_labels, pos_label: int = 1, ignore_label: int = -100):
	valid_mask = true_labels != ignore_label

	valid_true = true_labels[valid_mask] == pos_label
	valid_pred = pred_labels[valid_mask] == pos_label
	return recall_score(y_true=valid_true, y_pred=valid_pred, average="binary", pos_label=1)


def token_f1(true_labels, pred_labels, pos_label: int = 1, ignore_label: int = -100):
	valid_mask = true_labels != ignore_label

	valid_true = true_labels[valid_mask] == pos_label
	valid_pred = pred_labels[valid_mask] == pos_label
	return f1_score(y_true=valid_true, y_pred=valid_pred, average="binary", pos_label=1)


def preprocess_iob2(labels: List[str], fallback_label: str = "O") -> List[str]:
	""" Breaks up contiguous labels of the same type into beginning (B-)/inside(I-) labels. Keeps negative labels
	as they are. """
	pos, open_type = 0, None
	prepr_labels = []

	while pos < len(labels):
		if labels[pos] != fallback_label:
			if open_type is None:
				prepr_labels.append(f"B-{labels[pos]}")
				open_type = labels[pos]
			else:
				if labels[pos] == open_type:
					prepr_labels.append(f"I-{labels[pos]}")
				else:
					prepr_labels.append(f"B-{labels[pos]}")
					open_type = labels[pos]
		else:
			prepr_labels.append(labels[pos])
			open_type = None

		pos += 1

	return prepr_labels


if __name__ == "__main__":
	import torch

	t = torch.tensor([
		[-100, 1, 1, 1, 2, 2, 0, -100],
		[-100, 1, 0, 1, 2, 2, -100, -100],
	])
	p = torch.tensor([
		[-100, 2, 0, 1, 1, 2, 0, -100],
		[-100, 1, 0, 1, 2, 2, -100, -100],
	])

	print(token_precision(t, p, pos_label=1))
	print(token_recall(t, p, pos_label=1))
	print(token_f1(t, p, pos_label=1))
