'''Creates the training dataset for GPT
'''
import torch

PERCENT_TRAIN: float = 0.9 # 1 - PERCENT_TRAIN > 0.0


# Read the text
with open("data/input.txt", 'r', encoding='utf-8') as f:
    text = f.read()

# Info about dataset
# print("length of dataset in characters: ", len(text))

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


def tokenize_dataset(dataset_path: str) -> torch.Tensor:
    # Tokenize dataset
    data = torch.tensor(encode(text), dtype=torch.long)

    return data


if __name__ == '__main__': 
    data = tokenize_dataset("data/input.txt")

    # Split data into train and validation dataset
    n = int(PERCENT_TRAIN*len(data))
    train_data = data[:n]
    val_data = data[:n]     # Ensures that we don't overfit

    # Never feed entire text into transformer bc computation limits
    # Work with chunks of dataset (bits of the text)
    # Multiple examples packed in 
    # Want to predict all of the positions (ex. if 18 before, 47 most likely next)
    block_size = 8  # context-length
    print(train_data[:block_size+1])
    print(decode(train_data[:block_size+1].tolist()))   

    x = train_data[:block_size]
    y = train_data[1:block_size+1]
    # This is what we want the transformer to learn
    # This helps the transformer understand different context size
    # and predict character with context of just 1
    for t in range(block_size):
        context = x[:t+1]
        target = y[t]
        print(f"when input is {context} the target is {target}")
