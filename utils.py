from sklearn.metrics import precision_score, recall_score, f1_score


def token_precision(true_labels, pred_labels, pos_label: int = 1, ignore_label: int = -100):
	valid_mask = true_labels != ignore_label

	valid_true = true_labels[valid_mask] == pos_label
	valid_pred = pred_labels[valid_mask] == pos_label
	return precision_score(y_true=valid_true, y_pred=valid_pred, average="binary", pos_label=pos_label)


def token_recall(true_labels, pred_labels, pos_label: int = 1, ignore_label: int = -100):
	valid_mask = true_labels != ignore_label

	valid_true = true_labels[valid_mask] == pos_label
	valid_pred = pred_labels[valid_mask] == pos_label
	return recall_score(y_true=valid_true, y_pred=valid_pred, average="binary", pos_label=pos_label)


def token_f1(true_labels, pred_labels, pos_label: int = 1, ignore_label: int = -100):
	valid_mask = true_labels != ignore_label

	valid_true = true_labels[valid_mask] == pos_label
	valid_pred = pred_labels[valid_mask] == pos_label
	return f1_score(y_true=valid_true, y_pred=valid_pred, average="binary", pos_label=pos_label)


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
