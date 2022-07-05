from sklearn.metrics import precision_score, recall_score, f1_score, \
	average_precision_score, precision_recall_curve
import torch

from data import LOSS_IGNORE_INDEX


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


def token_average_precision(true_labels, pos_probas, ignore_label: int = -100):
	# pos_probas... predicted probability for the positive class (1)
	valid_mask = true_labels != ignore_label
	assert len(torch.unique(true_labels[valid_mask])) == 2, \
		"token_average_precision is implemented only for a binary task"

	return average_precision_score(true_labels[valid_mask], pos_probas[valid_mask])


def token_pr_curve(true_labels, pos_probas, ignore_label: int = -100):
	# pos_probas... predicted probability for the positive class (1)
	valid_mask = true_labels != ignore_label
	assert len(torch.unique(true_labels[valid_mask])) == 2, \
		"token_pr_curve is implemented only for a binary task"

	return precision_recall_curve(true_labels[valid_mask], pos_probas[valid_mask])


# TODO: define boilerplate here, use named_parameters for placeholders!!
VIS_BOILERPLATE = \
"""
<!DOCTYPE html>
<html>
<head>
	<meta charset='utf-8'>
	<meta name='viewport' content='width=device-width, initial-scale=1'>
	<title>Visualization</title>
	<style type='text/css'>
		.example {{
			margin-bottom: 20px;
		}}
		.labels {{
			display: flex;
			flex-direction: row;
			flex-wrap: wrap;
		}}
		.token {{
			padding: 3px;
			margin-right: 3px;
			max-width:100px;
			width: 100%;
		}}
		.annotation {{
			font-size: 10px;
		}}
	</style>
</head>
<body>
{formatted_examples}
</body>
</html>
"""


def visualize_token_predictions(tokens, token_predicted, token_true=None, uninteresting_labels=None):
	eff_true = token_true
	if token_true is None:
		eff_true = [None for _ in range(len(token_predicted))]
	assert len(token_predicted) == len(eff_true) == len(tokens)
	eff_uninteresting = set(uninteresting_labels) if uninteresting_labels is not None else {"O", "not_metaphor"}

	formatted_examples = []

	for curr_tokens, curr_preds, curr_true in zip(tokens, token_predicted, eff_true):
		if curr_true is None:
			curr_true = [None for _ in range(len(curr_preds))]
		assert len(curr_tokens) == len(curr_preds) == len(curr_true)
		formatted_examples.append("<div class='example'>")

		formatted_examples.append("<div class='metadata'>")
		formatted_examples.append("<code>{}</code>".format(" ".join(curr_tokens)))
		formatted_examples.append("</div>")

		formatted_examples.append("<div class='labels'>")
		for token, pred_lbl, true_lbl in zip(curr_tokens, curr_preds, curr_true):
			skip_token = pred_lbl == LOSS_IGNORE_INDEX or true_lbl == LOSS_IGNORE_INDEX
			if skip_token:
				continue

			border_color, bg_color = "#5c5b5b", "#e3e3e3"  # gray
			optional_tooltip = ""
			# Visualize correctness only for "interesting" labels, e.g., positive labels in metaphor detection
			if true_lbl is not None and (pred_lbl not in eff_uninteresting or true_lbl not in eff_uninteresting):
				if pred_lbl == true_lbl:
					border_color, bg_color = "#32a852", "#baffcd"  # green
				else:
					optional_tooltip = f" title='Correct: {true_lbl}'"
					border_color, bg_color = "#700900", "#ff9d94"  # red

			formatted_examples.append(f"<div class='token' style='border: 2px solid {border_color}; background-color: {bg_color};'{optional_tooltip}>")
			formatted_examples.append(f"<div style='text-align: center;' class='annotation'>{pred_lbl}</div>")
			formatted_examples.append(f"<div style='text-align: center;'>{token}</div>")
			formatted_examples.append("</div>")
		formatted_examples.append("</div>")

		formatted_examples.append("</div>")

	formatted_examples = "\n".join(formatted_examples)
	return VIS_BOILERPLATE.format(formatted_examples=formatted_examples)


if __name__ == "__main__":
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

	t_bin = torch.tensor([[-100, 0, 0, 1, 1, -100]])
	p_pos = torch.tensor([[0.5, 0.1, 0.4, 0.35, 0.8, 0.2]])
	print(token_average_precision(true_labels=t_bin, pos_probas=p_pos))  # 0.833.. (sklearn example)

	tokens = ['Moderen', ',', 'čist', 'in', 'preprost', '.']
	preds = ['O', 'O', 'MRWi', 'O', 'O', 'O']
	correct = ['O', 'O', 'MRWd', 'O', 'O', 'O']
	with open("tmp.html", "w", encoding="utf-8") as f:
		vis_html = visualize_token_predictions([tokens], [preds], [correct])
		print(vis_html, file=f)
