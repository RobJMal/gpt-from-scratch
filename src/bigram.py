'''Creates the training dataset for GPT
'''
# Third-party
import torch

# Custom
from model import BigramLanguageModel

# ---- Hyperparams ----
TORCH_SEED = 1337
PERCENT_TRAIN: float = 0.9 # 1 - PERCENT_TRAIN > 0.0
BATCH_SIZE: int = 32 # How many independent sequences will we process in parallel
BLOCK_SIZE: int = 8 # Maximum context length for predictions?
MAX_ITERS: int = 10000
EVAL_INTERVAL: int = 500
LEARNING_RATE: float = 1e-2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EVAL_ITERS: int = 200

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
val_data = data[:n]     # Ensures that we don't overfit

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

bigram_lm = BigramLanguageModel(vocab_size)
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
