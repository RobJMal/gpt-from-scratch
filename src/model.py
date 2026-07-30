import torch
import torch.nn as nn
from torch.nn import functional as F

TORCH_SEED = 1337

torch.manual_seed(TORCH_SEED)

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size, block_size, n_embed, device):
        super().__init__()
        self.block_size = block_size
        self.device = device

        # Each token directly readds of the lgits for next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        # Want to encode the position of the tokens as well, not just the identity
        self.position_embedding_table = nn.Embedding(block_size, n_embed)

        # Need linear layer to go from token-embeddings to logits
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets = None):
        B, T = idx.shape 
        # idx and targets are both (B, T) tensor of integers
        tokens_embed = self.token_embedding_table(idx)    # (Batch, Time, n_embed)
        position_embed = self.position_embedding_table(torch.arange(T, device=self.device)) # <T, C>
        x = tokens_embed + position_embed   # <B, T, C>, holds both the token identities and positions at which tokens occur
        logits = self.lm_head(x) # <B, T, vocab_size>

        # This is for some particular shape
        if targets is None:
            loss = None
        else: 
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            # Pytorch cross_entropy wants (B,C,T)
            loss = F.cross_entropy(logits, targets) # -log likelihood of cross and targets

        # Scores for next character in the sequence 
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop context idx to last block_size tokens so we don't go out of bounds
            idx_cond = idx[:, -self.block_size:]

            # get the predictions
            logits, loss = self(idx_cond)

            # focus on last time step
            logits = logits[:, -1, :]   # Becomes (B, C)

            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)   # (B, C)

            # sample from distribution, just 1 sample
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # append sampled indx to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

        return idx