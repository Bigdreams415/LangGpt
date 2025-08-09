import os
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase, NFD, Sequence

DATA_DIR = "../Dataset/processed"
TK_DIR = "../tokenizers"
os.makedirs(TK_DIR, exist_ok=True)

train_path = os.path.join(DATA_DIR, "train.csv")
val_path = os.path.join(DATA_DIR, "val.csv")
test_path = os.path.join(DATA_DIR, "test.csv")

train_out = os.path.join(DATA_DIR, "train_tokenized_hf.csv")
val_out = os.path.join(DATA_DIR, "validation_tokenized_hf.csv")
test_out = os.path.join(DATA_DIR, "test_tokenized_hf.csv")

eng_tok_path = os.path.join(TK_DIR, "english_tokenizer.json")
igb_tok_path = os.path.join(TK_DIR, "igbo_tokenizer.json")

def train_and_save_tokenizer(sentences, out_path):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.normalizer = Sequence([NFD(), Lowercase()])
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=4000, special_tokens=["[UNK]","[PAD]","[CLS]","[SEP]","[MASK]"])
    tokenizer.train_from_iterator(sentences, trainer)
    tokenizer.save(out_path)
    return tokenizer

if not (os.path.exists(eng_tok_path) and os.path.exists(igb_tok_path)):
    df = pd.read_csv(train_path)
    df["english"] = df["english"].astype(str).fillna("").str.strip()
    df["igbo"] = df["igbo"].astype(str).fillna("").str.strip()
    en_sents = df["english"].tolist()
    ig_sents = df["igbo"].tolist()
    train_and_save_tokenizer(en_sents, eng_tok_path)
    train_and_save_tokenizer(ig_sents, igb_tok_path)

tokenizer_en = Tokenizer.from_file(eng_tok_path)
tokenizer_ig = Tokenizer.from_file(igb_tok_path)

def tokenize_file(in_path, out_path):
    if not os.path.exists(in_path):
        print("skip missing:", in_path); return
    df = pd.read_csv(in_path)
    df["english"] = df["english"].astype(str).fillna("").str.strip()
    df["igbo"] = df["igbo"].astype(str).fillna("").str.strip()
    df["english_tokens"] = df["english"].apply(lambda x: tokenizer_en.encode(x).tokens)
    df["igbo_tokens"] = df["igbo"].apply(lambda x: tokenizer_ig.encode(x).tokens)
    df.to_csv(out_path, index=False)
    print("saved:", out_path, "rows:", len(df))

tokenize_file(train_path, train_out)
tokenize_file(val_path, val_out)
tokenize_file(test_path, test_out)
