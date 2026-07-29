'''Creates the training dataset for GPT
'''
import torch

PERCENT_TRAIN: float = 0.9 # 1 - PERCENT_TRAIN > 0.0

def tokenize_dataset(dataset_path: str) -> torch.Tensor:
    # Read the text
    with open(dataset_path, 'r', encoding='utf-8') as f:
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
    print(type(data))
    print(data.shape, data.dtype)
    print(data[:1000])

    return data


if __name__ == '__main__': 
    data = tokenize_dataset("data/input.txt")

    # Split data into train and validation dataset
    n = int(PERCENT_TRAIN*len(data))
    train_data = data[:n]
    val_data = data[:n]     # Ensures that we don't overfit

    # Never feed entire text into transformer bc computation limits
    # Work with chunks of dataset (bits of the text)
    block_size = 8  # context-length
    train_data[:block_size+1]
