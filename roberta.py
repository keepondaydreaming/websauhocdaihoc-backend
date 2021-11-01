import gc
from typing import Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig  # type: ignore
from transformers import AutoModel  # type: ignore
from transformers import AutoTokenizer  # type: ignore

gc.enable()

BATCH_SIZE = 32
MAX_LEN = 248
HIDDEN_STATE = 768
EVAL_SCHEDULE = [(0.50, 16), (0.49, 8), (0.48, 4), (0.47, 2), (-1.0, 1)]
ROBERTA_PATH = "roberta-base"
TOKENIZER_PATH = "roberta-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LitModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        config = AutoConfig.from_pretrained("roberta-base")
        config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": 0.0,
                "layer_norm_eps": 1e-7,
            }
        )

        self.roberta = AutoModel.from_config(config)  # type: ignore

        self.attention = nn.Sequential(
            nn.Linear(HIDDEN_STATE, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1),
        )

        self.regressor = nn.Sequential(nn.Linear(HIDDEN_STATE, 1))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> nn.Module:
        roberta_output = self.roberta(
            input_ids=input_ids, attention_mask=attention_mask
        )

        # There are a total of 13 layers of hidden states.
        # 1 for the embedding layer, and 12 for the 12 Roberta layers.
        # We take the hidden states from the last Roberta layer.
        last_layer_hidden_states = roberta_output.hidden_states[-1]

        # The number of cells is MAX_LEN.
        # The size of the hidden state of each cell is 768 (for roberta-base).
        # In order to condense hidden states of all cells to a context vector,
        # we compute a weighted average of the hidden states of all cells.
        # We compute the weight of each cell, using the attention neural network.
        weights = self.attention(last_layer_hidden_states)

        # weights.shape is BATCH_SIZE x MAX_LEN x 1
        # last_layer_hidden_states.shape is BATCH_SIZE x MAX_LEN x 768
        # Now we compute context_vector as the weighted average.
        # context_vector.shape is BATCH_SIZE x 768
        context_vector = torch.sum(weights * last_layer_hidden_states, dim=1)

        # Now we reduce the context vector to the prediction score.
        return self.regressor(context_vector)


class Inference:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        self._load_model()

    def _load_model(self) -> None:
        print(f"\nUsing {self.model_path}")

        self.model = LitModel()
        self.model.load_state_dict(torch.load(self.model_path, map_location=DEVICE))
        self.model.to(DEVICE)
        self.model.eval()

    def _preprocess(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            text,
            padding="max_length",
            max_length=MAX_LEN,
            truncation=True,
            return_attention_mask=True,
        )
        input_ids = torch.tensor(encoded['input_ids'])
        attention_mask = torch.tensor(encoded['attention_mask'])
        return (input_ids, attention_mask)

    def predict(self, excerpt: str) -> float:
        with torch.no_grad():
            input_ids, attention_mask = self._preprocess(excerpt)

            input_ids = input_ids.to(DEVICE).unsqueeze(0)
            attention_mask = attention_mask.to(DEVICE).unsqueeze(0)

            score = self.model(input_ids, attention_mask)

            return score.cpu().item()
