import logging
import os

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


# Adapted code from:
# https://towardsdatascience.com/how-to-create-and-train-a-multi-task-transformer-model-18c54a146240
class AutoModelForTokenMultiClassification(nn.Module):
	def __init__(self, pretrained_name_or_path, num_types: int, num_frames: int):
		super().__init__()

		self.encoder = AutoModel.from_pretrained(pretrained_name_or_path)
		self.hidden_size = self.encoder.config.hidden_size
		self.num_types = num_types
		self.num_frames = num_frames
		self.dropout_p = (
			self.encoder.config.classifier_dropout
			if self.encoder.config.classifier_dropout is not None else self.encoder.config.hidden_dropout_prob
		)

		self.output_heads = nn.ModuleDict({
			"metaphor_type": TokenClassificationHead(self.hidden_size, self.num_types, dropout_p=self.dropout_p,
													 sd=self.encoder.config.initializer_range),
			"metaphor_frame": TokenClassificationHead(self.hidden_size, self.num_frames, dropout_p=self.dropout_p,
													  sd=self.encoder.config.initializer_range)
		})

		# Load pre-trained weights for output_heads if they exist in model dir
		self.metaphor_type_path = os.path.join(pretrained_name_or_path, "metaphor_type.th")
		if os.path.exists(self.metaphor_type_path):
			logging.info(f"Loading weights for 'metaphor_type' token classification head from '{self.metaphor_type_path}'")
			self.output_heads["metaphor_type"].load_state_dict(torch.load(self.metaphor_type_path))

		self.metaphor_frame_path = os.path.join(pretrained_name_or_path, "metaphor_frame.th")
		if os.path.exists(self.metaphor_frame_path):
			logging.info(f"Loading weights for 'metaphor_frame' token classification head from '{self.metaphor_frame_path}'")
			self.output_heads["metaphor_frame"].load_state_dict(torch.load(self.metaphor_frame_path))

	def save_pretrained(self, model_dir):
		self.encoder.save_pretrained(model_dir)
		torch.save(self.output_heads["metaphor_type"].state_dict(), os.path.join(model_dir, "metaphor_type.th"))
		torch.save(self.output_heads["metaphor_frame"].state_dict(), os.path.join(model_dir, "metaphor_frame.th"))

	def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
				position_ids=None, head_mask=None, inputs_embeds=None, labels=None,
				**kwargs):
		metaphor_types = labels
		metaphor_frames = kwargs.get("frame_labels", None)

		outputs = self.encoder(
			input_ids=input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
			position_ids=position_ids,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
		)

		sequence_output = outputs[0]
		loss_list = []
		logits = {}

		for curr_task, ground_truth in [("metaphor_type", metaphor_types), ("metaphor_frame", metaphor_frames)]:
			mtype_logits, mtype_loss = self.output_heads[curr_task].forward(
				sequence_output,
				labels=None if ground_truth is None else ground_truth,
				attention_mask=attention_mask,
			)
			logits[curr_task] = mtype_logits
			if ground_truth is not None:
				loss_list.append(mtype_loss)

		ret_dict = {"logits": logits}
		if loss_list:
			ret_dict["loss"] = torch.stack(loss_list).mean()

		return ret_dict


class TokenClassificationHead(nn.Module):
	def __init__(self, hidden_size, num_labels, dropout_p=0.1, sd=0.02):
		super().__init__()
		self.dropout = nn.Dropout(dropout_p)
		self.classifier = nn.Linear(hidden_size, num_labels)
		self.num_labels = num_labels
		self.sd = sd

		self._init_weights()

	def _init_weights(self):
		self.classifier.weight.data.normal_(mean=0.0, std=self.sd)
		if self.classifier.bias is not None:
			self.classifier.bias.data.zero_()

	def forward(
			self, sequence_output, labels=None, attention_mask=None, **kwargs
	):
		sequence_output_dropout = self.dropout(sequence_output)
		logits = self.classifier(sequence_output_dropout)

		loss = None
		if labels is not None:
			loss_fct = torch.nn.CrossEntropyLoss()

			labels = labels.long()

			# Only keep active parts of the loss
			if attention_mask is not None:
				active_loss = attention_mask.view(-1) == 1
				active_logits = logits.view(-1, self.num_labels)
				active_labels = torch.where(
					active_loss,
					labels.view(-1),
					torch.tensor(loss_fct.ignore_index).type_as(labels),
				)
				loss = loss_fct(active_logits, active_labels)
			else:
				loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

		return logits, loss


if __name__ == "__main__":
	tokenizer = AutoTokenizer.from_pretrained("EMBEDDIA/sloberta")
	model = AutoModelForTokenMultiClassification("EMBEDDIA/sloberta", num_types=2, num_frames=5)

	res = model(**tokenizer(["Bagnolet je vzhodno predmestje Pariza"], return_tensors="pt"))
	print(res)

