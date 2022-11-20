from copy import deepcopy

import torch
from sklearn.metrics import precision_score, recall_score, f1_score, \
	average_precision_score, precision_recall_curve


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
		
		.square {{
			height: 16px;
			width: 16px;
			background-color: rgba(200, 200, 200, 1);
			margin-bottom: -3px;
			display: inline-block;
			border: 1px solid black;
			border-radius: 3px;
		}}

		.example {{
			word-break: break-word;
			padding-top: 10px;
			padding-left: 5%;
			padding-right: 5%;
			padding-bottom: 10px;
		}}

		.example-text {{
			padding: 5px;
		}}

		.example-label {{
			font-size: 18px;
		}}
	</style>
</head>
<body>
{formatted_examples}
</body>
</html>
"""


def visualize_predictions(sentences,
						  ysent_pred=None, ysent_true=None,
						  ytoken_pred=None, ytoken_true=None):
	visualize_sent = ysent_pred is not None
	visualize_tok = ytoken_pred is not None
	assert visualize_sent or visualize_tok, f"Either sentence-level or token-level predictions (or both) must be provided"
	num_ex = len(sentences)

	eff_ysent_true = ysent_true
	if ysent_true is None:
		eff_ysent_true = [None for _ in range(num_ex)]

	eff_ytoken_true = ytoken_true
	if ytoken_true is None:
		eff_ytoken_true = [None for _ in range(num_ex)]

	eff_ysent_pred = ysent_pred
	if ysent_pred is None:
		eff_ysent_pred = [None for _ in range(num_ex)]

	eff_ytoken_pred = ytoken_pred
	if ytoken_pred is None:
		eff_ytoken_pred = [None for _ in range(num_ex)]

	assert len(eff_ysent_true) == len(eff_ytoken_true) == len(eff_ysent_pred) == len(eff_ytoken_pred)

	formatted_examples = []
	for words, sent_pred, sent_true, tok_pred, tok_true in zip(sentences,
															   eff_ysent_pred, eff_ysent_true,
															   eff_ytoken_pred, eff_ytoken_true):
		formatted_examples.append("<div class='example'>")

		# Sentence prediction visualization
		if sent_pred is not None:
			aux_correct_text = ""
			color = "#e3e3e3"
			if sent_true is not None:
				if sent_pred == sent_true:
					color = "#baffcd"
				else:
					color = "#ff9d94"
					aux_correct_text = f" (<i>correct: {sent_true}</i>)"

			formatted_examples.append(
				f"<div>"
				f"<span style='background-color: {color}'>"
				f"<strong>Predicted:</strong> {sent_pred}"
				f"{aux_correct_text}"
				f"</span>"
				f"</div>")

		# Token prediction visualization
		if tok_pred is not None or tok_true is not None:
			# First visualize highlighted token predictions
			if tok_pred is not None:
				assert len(tok_pred) == len(words)

				pred_marked_words = deepcopy(words)

				for _i, _pred in enumerate(tok_pred):
					if _pred != "O" and _pred != 0:
						pred_marked_words[_i] = f"<span style='background-color: #c1c7c9; padding: 2px'>{pred_marked_words[_i]}</span>"

				formatted_examples.append("<div><strong>Token predictions:</strong><br />{}</div>".format(
					" ".join(pred_marked_words)
				))

			# Then visualize highlighted token correct annotations
			if tok_true is not None:
				assert len(tok_true) == len(words)
				true_marked_words = deepcopy(words)

				for _i, _class in enumerate(tok_true):
					if _class not in {"O", "not_metaphor", 0}:
						true_marked_words[_i] = f"<span style='background-color: #c1c7c9; padding: 2px' title={_class}>" \
												f"{true_marked_words[_i]}" \
												f"</span>"

				formatted_examples.append("<div><strong>Human annotations:</strong><br />{}</div>".format(" ".join(true_marked_words)))
		else:
			# If there are no token annotations provided, simply output the sentence words
			formatted_examples.append("<div><strong>Input sentence:</strong><br />{}</div>".format(" ".join(words)))

		formatted_examples.append("</div>")

	formatted_examples = "\n".join(formatted_examples)
	return VIS_BOILERPLATE.format(formatted_examples=formatted_examples)


if __name__ == "__main__":
	from data import load_df

	data_paths = [
		"RESULTS/SENTENCE/komet-sent-csebert-2e-5-binary3-history0-optthresh/komet-sent-fold0-csebert-2e-5-binary3-history0-optthresh/predict-test/test_results.tsv"
	]

	# Token visualization logic
	# all_words, all_token_preds, all_token_true = [], [], []
	# for curr_path in data_paths:
	# 	df = load_df(curr_path)
	#
	# 	all_words.extend(df["sentence_words"].tolist())
	# 	all_token_preds.extend(df["preds_transformed"].tolist())
	# 	all_token_true.extend(df["true_transformed"].tolist())
	#
	# with open("token_visualization.html", "w", encoding="utf-8") as f:
	# 	vis_html = visualize_predictions(all_words, ytoken_pred=all_token_preds, ytoken_true=all_token_true)
	# 	print(vis_html, file=f)

	# Sentence visualization logic
	all_words, all_sent_preds, all_sent_true, all_token_true = [], [], [], []
	for curr_path in data_paths:
		df = load_df(curr_path)

		all_words.extend(df["sentence_words"].tolist())
		all_sent_preds.extend(df["preds_transformed"].tolist())
		all_sent_true.extend(df["true_transformed"].tolist())

		token_true_dense = []
		for words, met_type in zip(df["sentence_words"].tolist(),
								   df["met_type"].tolist()):
			curr_dense = ["O" for _ in range(len(words))]
			for met_info in met_type:
				for _i in met_info["word_indices"]:
					curr_dense[_i] = met_info["type"]

			token_true_dense.append(curr_dense)
		all_token_true.extend(token_true_dense)

	with open("sentence_visualization.html", "w", encoding="utf-8") as f:
		vis_html = visualize_predictions(all_words, ysent_pred=all_sent_preds, ysent_true=all_sent_true,
										 ytoken_true=all_token_true)
		print(vis_html, file=f)
