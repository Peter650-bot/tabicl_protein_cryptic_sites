"""Class definitions the published cryptic-site model was pickled with.

Copied verbatim (classes only) from skrhakv/cryptic-finetuning,
src/finetuning_utils.py -- MIT License, (c) Vít Škrhák. `cbs-model.pt` is a whole
pickled `nn.Module`, so `torch.load` needs these classes importable under the
module name recorded in the pickle; `finetuned_esm_modal.py` registers this file
under that name. Nothing here is modified: a changed forward() would silently
score a different model than the one the papers report.
"""
import numpy as np
import torch
import torch.nn as nn
from transformers import EsmModel

DROPOUT = 0.3
OUTPUT_SIZE = 1
NUMBER_OF_MODES = 15
OUT_CHANNELS = 128
KERNEL_SIZE = 15
NUM_LAYERS = 3


class FinetunedEsmModelWithCnn(nn.Module):
    def __init__(self, esm_model: str, in_channels:int=1) -> None:
        super().__init__()
        self.llm = EsmModel.from_pretrained(esm_model) #, torch_dtype=torch.bfloat16)

        conv_layers = [nn.Conv1d(in_channels, OUT_CHANNELS, KERNEL_SIZE, padding=KERNEL_SIZE // 2),
                      nn.ReLU()]
        for i in range(NUM_LAYERS - 1):
            conv_layers.append(nn.Conv1d(OUT_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, padding=KERNEL_SIZE // 2))
            conv_layers.append(nn.ReLU())
            # conv_layers.append(nn.BatchNorm1d(OUT_CHANNELS))
        self.pLDDT_conv = nn.Sequential(*conv_layers)

        self.dropout = nn.Dropout(DROPOUT)

        input_size = self.llm.config.hidden_size + OUT_CHANNELS
        self.classifier = nn.Linear(input_size, OUTPUT_SIZE) # , dtype=torch.bfloat16)

    def forward(self, batch: dict[str, np.ndarray]) -> torch.Tensor:
        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
        if 'plDDTs' in batch:
            pLDDT = batch["plDDTs"].half()
        else:
            pLDDT = []
            for i in range(NUMBER_OF_MODES):
                pLDDT.append(batch[f'nma_mode_{i+1}'].half())
            pLDDT = torch.stack(pLDDT, dim=2)  #

        token_embeddings = self.llm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        token_embeddings = self.dropout(token_embeddings)

        pLDDT = pLDDT.reshape(token_embeddings.shape[0], -1, token_embeddings.shape[1])
        processed_pLDDT = self.pLDDT_conv(pLDDT)
        processed_pLDDT = processed_pLDDT.reshape(token_embeddings.shape[0], token_embeddings.shape[1], OUT_CHANNELS)
        assert processed_pLDDT.shape[0] == token_embeddings.shape[0]
        assert processed_pLDDT.shape[1] == token_embeddings.shape[1]

        token_embeddings = torch.cat((token_embeddings, processed_pLDDT), dim=2)

        assert processed_pLDDT.shape[1] == token_embeddings.shape[1]
        return self.classifier(token_embeddings)


class MultitaskFinetunedEsmModelWithCnn(nn.Module):
    def __init__(self, esm_model: str) -> None:
        super().__init__()
        self.llm = EsmModel.from_pretrained(esm_model) #, torch_dtype=torch.bfloat16)

        conv_layers = [nn.Conv1d(1, OUT_CHANNELS, KERNEL_SIZE, padding=KERNEL_SIZE // 2),
                      nn.ReLU()]
        for i in range(NUM_LAYERS - 1):
            conv_layers.append(nn.Conv1d(OUT_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, padding=KERNEL_SIZE // 2))
            conv_layers.append(nn.ReLU())
            # conv_layers.append(nn.BatchNorm1d(OUT_CHANNELS))
        self.pLDDT_conv = nn.Sequential(*conv_layers)

        self.dropout = nn.Dropout(DROPOUT)

        input_size = self.llm.config.hidden_size + OUT_CHANNELS
        self.classifier = nn.Linear(input_size, OUTPUT_SIZE) # , dtype=torch.bfloat16)
        self.plDDT_regressor = nn.Linear(input_size, OUTPUT_SIZE) # , dtype=torch.bfloat16)
        self.distance_regressor = nn.Linear(input_size, OUTPUT_SIZE) # , dtype=torch.bfloat16)

    def forward(self, batch: dict[str, np.ndarray]) -> torch.Tensor:
        input_ids, attention_mask, pLDDT = batch["input_ids"], batch["attention_mask"], batch["plDDTs"].half()

        token_embeddings = self.llm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        token_embeddings = self.dropout(token_embeddings)

        pLDDT = pLDDT.unsqueeze(1)
        processed_pLDDT = self.pLDDT_conv(pLDDT)
        processed_pLDDT = processed_pLDDT.reshape(token_embeddings.shape[0], token_embeddings.shape[1], OUT_CHANNELS)
        assert processed_pLDDT.shape[0] == token_embeddings.shape[0]
        assert processed_pLDDT.shape[1] == token_embeddings.shape[1]

        token_embeddings = torch.cat((token_embeddings, processed_pLDDT), dim=2)

        assert processed_pLDDT.shape[1] == token_embeddings.shape[1]
        return self.classifier(token_embeddings), self.plDDT_regressor(token_embeddings), self.distance_regressor(token_embeddings)

class MultitaskFinetunedEsmModel(nn.Module):
    def __init__(self, esm_model: str) -> None:
        super().__init__()
        self.llm = EsmModel.from_pretrained(esm_model)
        self.dropout = nn.Dropout(DROPOUT)
        self.classifier = nn.Linear(self.llm.config.hidden_size, OUTPUT_SIZE)
        self.plDDT_regressor = nn.Linear(self.llm.config.hidden_size, OUTPUT_SIZE)
        self.distance_regressor = nn.Linear(self.llm.config.hidden_size, OUTPUT_SIZE)

    def forward(self, batch: dict[str, np.ndarray]) -> torch.Tensor:
        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
        token_embeddings = self.llm(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state

        return self.classifier(token_embeddings), self.plDDT_regressor(token_embeddings), self.distance_regressor(token_embeddings)

class FinetunedEsmModel(nn.Module):
    def __init__(self, esm_model: str) -> None:
        super().__init__()
        self.llm = EsmModel.from_pretrained(esm_model)

        self.dropout = nn.Dropout(DROPOUT)
        self.classifier = nn.Linear(self.llm.config.hidden_size, OUTPUT_SIZE)

    def forward(self, batch: dict[str, np.ndarray]) -> torch.Tensor:
        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
        token_embeddings = self.llm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        return self.classifier(token_embeddings)
