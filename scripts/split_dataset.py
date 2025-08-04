import pandas as pd
from sklearn.model_selection import train_test_split
import os


# path
raw_file = "Dataset/raw/igbo_english_full.csv"
processed_dir ="Dataset/processed"

# Load the dataset
df = pd.read_csv(raw_file)

# Split the dataset into training, validation, and test sets/shuffle the dataset

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
# Split the dataset into train(80%), temp(20%)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
# Split the temp set into validation(10%) and test(10%)
val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)

os.makedirs(processed_dir, exist_ok=True)

# Saving my datasets
train_df.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
val_df.to_csv(os.path.join(processed_dir, "val.csv"), index=False)
test_df.to_csv(os.path.join(processed_dir, "test.csv"), index=False)

print("Dataset successfully split into train, validation, and test sets.")
print(f"Train set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Test set size: {len(test_df)}")
print("Files saved in:", processed_dir)
