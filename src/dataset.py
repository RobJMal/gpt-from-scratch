'''Creates the training dataset for GPT
'''
import torch

# Read the text
with open("data/input.txt", 'r', encoding='utf-8') as f:
    text = f.read()

# Info about dataset
print("length of dataset in characters: ", len(text))

# Extracting unique characters that occur in text
chars = sorted(list(set(text))) # want sorting to have consistent
vocab_size = len(chars)

print("tokens of dataset: ", chars)
print("number of unique tokens: ", vocab_size)

# --- TOKENIZER ---
# Bc building a character-level language model, tokenizer is tokens on
# the characters
ctoi: dict[str, int] = { ch:i for i,ch in enumerate(chars) }    # Convert char->int
itoc: dict[int, str] = { i:ch for i,ch in enumerate(chars) }    # Convert int->char

# Functions that do encoding and decoding
encode = lambda s: [ctoi[c] for c in s]     # encoder: string -> list of ints
decode = lambda l: ''.join(itoc[i] for i in l)  # decoder: list of ints -> string

print(encode("hi robot"))
print(decode(encode("hi robot")))

# Tokenize dataset
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])