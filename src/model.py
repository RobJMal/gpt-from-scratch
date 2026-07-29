import torch
import torch.nn as nn
from torch.nn import functional as F

TORCH_SEED = 1337

torch.manual_seed(TORCH_SEED)

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        # Each token directly readds of the lgits for next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets):
        # idx and targets are both (B, T) tensor of integers
        logits = self.token_embedding_table(idx)    # (Batch, Time, Channel)

        # Scores for next character in the sequence 
        return logits