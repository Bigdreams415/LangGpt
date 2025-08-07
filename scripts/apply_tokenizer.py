import pandas as pd
from tokenizers import Tokenizer
import os

# Load dataset
data = pd.read_csv("Dataset/processed/train.csv")
data["english"] = data["english"].astype(str).fillna("")
data["igbo"] = data["igbo"].astype(str).fillna("")

# Load trained tokenizers
tokenizer_en = Tokenizer.from_file("tokenizers/english_tokenizer.json")
tokenizer_ig = Tokenizer.from_file("tokenizers/igbo_tokenizer.json")

# Apply tokenization
print("Tokenizing English sentences...")
data["english_tokens"] = data["english"].apply(lambda x: tokenizer_en.encode(x).tokens)

print("Tokenizing Igbo sentences...")
data["igbo_tokens"] = data["igbo"].apply(lambda x: tokenizer_ig.encode(x).tokens)

# Save output
output_path = "Dataset/processed/train_tokenized_hf.csv"
data.to_csv(output_path, index=False)
print(f"✅ Tokenized data saved to {output_path}")
