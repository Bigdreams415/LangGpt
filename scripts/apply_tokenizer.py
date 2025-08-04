import pandas as pd
from bpe_tokenizer import BPETokenizer

# Load dataset
data = pd.read_csv("Dataset/processed/train.csv")
english_sentences = data["english"].astype(str).tolist()
igbo_sentences = data["igbo"].astype(str).tolist()

# Train English tokenizer
print("Training tokenizer on English sentences...")
tokenizer_en = BPETokenizer(vocab_size=1000)
tokenizer_en.train(english_sentences)

# Train Igbo tokenizer
print("Training tokenizer on Igbo sentences...")
tokenizer_ig = BPETokenizer(vocab_size=1000)
tokenizer_ig.train(igbo_sentences)

# Tokenize sentences
print("Encoding English sentences...")
data["english_tokens"] = [tokenizer_en.encode_sentence(text) for text in english_sentences]

print("Encoding Igbo sentences...")
data["igbo_tokens"] = [tokenizer_ig.encode_sentence(text) for text in igbo_sentences]

# Save to CSV
data.to_csv("Dataset/processed/train_tokenized.csv", index=False)
print("✅ Tokenized data saved to Dataset/processed/train_tokenized.csv")
