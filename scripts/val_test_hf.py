import os
import pandas as pd
from tokenizers import Tokenizer

# Paths
tokenizer_dir = "tokenizers"
validation_path = "Dataset/processed/val.csv"
test_path = "Dataset/processed/test.csv"

val_output_path = "Dataset/processed/validation_tokenized_hf.csv"
test_output_path = "Dataset/processed/test_tokenized_hf.csv"

# Load trained tokenizers
tokenizer_en = Tokenizer.from_file(os.path.join(tokenizer_dir, "english_tokenizer.json"))
tokenizer_ig = Tokenizer.from_file(os.path.join(tokenizer_dir, "igbo_tokenizer.json"))

def tokenize_and_save(input_path, output_path, tokenizer_en, tokenizer_ig):
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    data = pd.read_csv(input_path)
    data["english"] = data["english"].astype(str).fillna("")
    data["igbo"] = data["igbo"].astype(str).fillna("")

    print(f"Tokenizing: {input_path}")
    data["english_tokens"] = data["english"].apply(lambda x: tokenizer_en.encode(x).tokens)
    data["igbo_tokens"] = data["igbo"].apply(lambda x: tokenizer_ig.encode(x).tokens)

    data.to_csv(output_path, index=False)
    print(f"Saved tokenized data to {output_path}")

# Tokenize validation and test (if test exists)
tokenize_and_save(validation_path, val_output_path, tokenizer_en, tokenizer_ig)
tokenize_and_save(test_path, test_output_path, tokenizer_en, tokenizer_ig)
