import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase, NFD, Sequence
import os

# Paths
train_path = "Dataset/processed/train.csv"
output_path = "Dataset/processed/train_tokenized_hf.csv"
tokenizer_dir = "tokenizers"
os.makedirs(tokenizer_dir, exist_ok=True)

# Load data
data = pd.read_csv(train_path)

# Fill NaN with empty strings
data["english"] = data["english"].astype(str).fillna("")
data["igbo"] = data["igbo"].astype(str).fillna("")

english_sentences = data["english"].tolist()
igbo_sentences = data["igbo"].tolist()

def train_and_save_tokenizer(sentences, lang_code):
    print(f"Training {lang_code} tokenizer...")
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    # Keep diacritics: only decompose (NFD) and lowercase
    tokenizer.normalizer = Sequence([NFD(), Lowercase()])
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=4000,
        special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    )
    tokenizer.train_from_iterator(sentences, trainer)
    path = os.path.join(tokenizer_dir, f"{lang_code}_tokenizer.json")
    tokenizer.save(path)
    print(f"Saved {lang_code} tokenizer to {path}")
    return tokenizer

# Train English and Igbo tokenizers
tokenizer_en = train_and_save_tokenizer(english_sentences, "english")
tokenizer_ig = train_and_save_tokenizer(igbo_sentences, "igbo")

# Tokenize and store tokens
print("Tokenizing dataset...")
data["english_tokens"] = data["english"].apply(lambda x: tokenizer_en.encode(x).tokens)
data["igbo_tokens"] = data["igbo"].apply(lambda x: tokenizer_ig.encode(x).tokens)

# Save to CSV
data.to_csv(output_path, index=False)
print(f"Tokenized data saved to {output_path}")
