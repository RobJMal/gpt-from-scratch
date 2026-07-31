'''Creates the training dataset for GPT
'''
# Third-party
import torch
import torch.nn as nn
from torch.nn import functional as F

# ---- Hyperparams ----
TORCH_SEED = 1337
PERCENT_TRAIN: float = 0.9 # 1 - PERCENT_TRAIN > 0.0
BATCH_SIZE: int = 32 # How many independent sequences will we process in parallel
BLOCK_SIZE: int = 8 # Maximum context length for predictions?
MAX_ITERS: int = 10000
EVAL_INTERVAL: int = 500
LEARNING_RATE: float = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EVAL_ITERS: int = 200
N_EMBED: int = 32    # Number of embedding dimensions


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBED, head_size, bias=False)
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)

        # This creates the register buffer
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)     # <B, T, C>
        q = self.query(x)   # <B, T, C>
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5     # <B, T, C> @ <B, C, T> --> <B, T, T>
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))    # <B, T, T>
        wei = F.softmax(wei, dim=-1)    # <B, T, T>
        # perform weighted aggregation of the values
        v = self.value(x)   # <B, T, C>
        out = wei @ v   # <B, T, T> @ <B, T, C> --> <B, T, C>
        return out


class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # Each token directly reads of the logits for next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBED)
        # Want to encode the position of the tokens as well, not just the identity
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBED)

        self.sa_head = Head(N_EMBED)
        # Need linear layer to go from token-embeddings to logits
        self.lm_head = nn.Linear(N_EMBED, vocab_size)

    def forward(self, idx, targets = None):
        B, T = idx.shape 

        # idx and targets are both (B, T) tensor of integers
        tokens_embed = self.token_embedding_table(idx)    # (Batch, Time, n_embed)
        position_embed = self.position_embedding_table(torch.arange(T, device=DEVICE)) # <T, C>

        x = tokens_embed + position_embed   # <B, T, C>, holds both the token identities and positions at which tokens occur
        x = self.sa_head(x)    # Apply one head of self-attention <B, T, C>
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
            # This is because we're using positional embeddings
            idx_cond = idx[:, -BLOCK_SIZE:]

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


# Read the text
with open("data/input.txt", 'r', encoding='utf-8') as f:
    text = f.read()

# Extracting unique characters that occur in text
chars = sorted(list(set(text))) # want sorting to have consistent
vocab_size = len(chars)

# --- TOKENIZER ---
# Bc building a character-level language model, tokenizer is tokens on
# the characters
ctoi: dict[str, int] = { ch:i for i,ch in enumerate(chars) }    # Convert char->int
itoc: dict[int, str] = { i:ch for i,ch in enumerate(chars) }    # Convert int->char
# Functions that do encoding and decoding
encode = lambda s: [ctoi[c] for c in s]     # encoder: string -> list of ints
decode = lambda l: ''.join(itoc[i] for i in l)  # decoder: list of ints -> string

# Split data into train and validation dataset
data = torch.tensor(encode(text), dtype=torch.long)
n = int(PERCENT_TRAIN*len(data))
train_data = data[:n]
val_data = data[n:]     # Ensures that we don't overfit

# Never feed entire text into transformer bc computation limits
# Work with chunks of dataset (bits of the text)
# Multiple examples packed in 
# Want to predict all of the positions (ex. if 18 before, 47 most likely next)
print(train_data[:BLOCK_SIZE+1])
print(decode(train_data[:BLOCK_SIZE+1].tolist()))   

x = train_data[:BLOCK_SIZE]
y = train_data[1:BLOCK_SIZE+1]
# This is what we want the transformer to learn
# This helps the transformer understand different context size
# and predict character with context of just 1
for t in range(BLOCK_SIZE):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target is {target}")

# Taking care of batch dimensions to keep GPUs busy
def get_batch(split: str):
    # Generates small batch of data of inputs x and targets y
    # Size is BATCH_SIZE x BLOCK_SIZE
    data = train_data if split == 'train' else val_data

    # Generate random positions to grab chunk off of
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))

    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    # Averages up the loss over multiple batches
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)
print('----')

for b in range(BATCH_SIZE):
    for t in range(BLOCK_SIZE):
        context = xb[b, :t+1]
        target = yb[b, t]
        print(f"when input is '{decode(context.tolist())}' the target is '{decode([target.tolist()])}'")

bigram_lm = BigramLanguageModel()
bigram_lm = bigram_lm.to(DEVICE)
logits, loss = bigram_lm(xb, yb) # Passing inputs and targets
print(logits.shape)
print(loss)

print()
context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE) # Starts off the generation
print(decode(bigram_lm.generate(context, max_new_tokens=100)[0].tolist()))

# PyTorch optimizer
optimizer = torch.optim.AdamW(bigram_lm.parameters(), lr=1e-3)

print("Running training...")
for iter in range(MAX_ITERS):

    # every once in a while evaluate loss on train and val sets
    if iter % EVAL_INTERVAL == 0:
        losses = estimate_loss(bigram_lm)
        print(f"step {iter}: train los {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = bigram_lm(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE) # Starts off the generation
print(decode(bigram_lm.generate(context, max_new_tokens=100)[0].tolist()))
