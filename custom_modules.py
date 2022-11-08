import os
from typing import Optional, List

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification


class TokenClassifierWithLayerCombination(nn.Module):
	def __init__(self, pretrained_name_or_path, **transformers_kwargs):
		super().__init__()
		self.model = AutoModelForTokenClassification.from_pretrained(pretrained_name_or_path, output_hidden_states=True,
																	 **transformers_kwargs)
		self.train()
		self.num_layers = self.model.config.num_hidden_layers
		# Initialize layer weights uniformly
		self.layer_combination = nn.Parameter(torch.ones((self.num_layers, 1, 1, 1), dtype=torch.float32) / self.num_layers,
											  requires_grad=self.training)

	def forward(self, **model_inputs):
		res = self.model(**model_inputs)

		# Custom pooling: weighted sum with learned layer weights
		hidden_states = torch.stack(res.hidden_states[-self.num_layers:])  # [num_layers, num_tokens, hidden_size]
		hidden_combo = torch.sum(torch.softmax(self.layer_combination, dim=0) * hidden_states, dim=0)  # [num_tokens, hidden_size]

		hidden_combo = self.model.dropout(hidden_combo)
		logits = self.model.classifier(hidden_combo)

		ret_dict = {"logits": logits}
		labels: Optional[torch.Tensor] = model_inputs.get("labels", None)
		if labels is not None:
			loss_fct = CrossEntropyLoss()
			loss = loss_fct(logits.view(-1, self.model.num_labels), labels.view(-1))
			ret_dict["loss"] = loss

		return ret_dict

	def save_pretrained(self, save_dir):
		self.model.save_pretrained(save_dir)
		torch.save(self.layer_combination.data, os.path.join(save_dir, "layer_combination.th"))

	@staticmethod
	def from_pretrained(model_dir, **transformers_kwargs):
		model = TokenClassifierWithLayerCombination(model_dir, **transformers_kwargs)
		if os.path.exists(model_dir):
			# Override uniform weights with learned ones
			model.layer_combination.data = torch.load(os.path.join(model_dir, "layer_combination.th"))

		return model


if __name__ == "__main__":
	tokenizer = AutoTokenizer.from_pretrained("EMBEDDIA/sloberta")
	model = TokenClassifierWithLayerCombination("EMBEDDIA/sloberta")
	model.train()

	for param_name, param_group in model.named_parameters():
		print(param_name)

	sentence = ["Tokrat", "se", "ni", "prvič", "srečal", "s", "projektom", "hujšanja"]
	encoded = tokenizer.encode_plus(sentence, is_split_into_words=True, return_tensors="pt")

	res = model(**encoded)
