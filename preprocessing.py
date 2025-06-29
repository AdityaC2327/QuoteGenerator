import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import torch

# Load dataset
df = pd.read_csv("quotes_by_mood.csv")

# Step 1: Clean the data
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.replace("\n", " ")  # Remove newlines
    text = text.strip()  # Trim spaces
    return text

df["Cleaned_Quote"] = df["quote"].apply(clean_text)

# Step 2: Format data for GPT-2
df["input_text"] = df["mood"] + ": " + df["Cleaned_Quote"]

# Step 3: Train-Validation Split
train_texts, val_texts = train_test_split(df[["input_text"]], test_size=0.2, random_state=42)

# Load GPT-2 Tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set EOS token for padding

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["input_text"], truncation=True, padding="max_length", max_length=128)

# Convert Pandas DataFrame to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_texts).map(tokenize_function, batched=True)
val_dataset = Dataset.from_pandas(val_texts).map(tokenize_function, batched=True)

# Save processed datasets (optional)
train_dataset.save_to_disk("train_dataset")
val_dataset.save_to_disk("val_dataset")

print("Preprocessing complete! Train & validation datasets are ready.")
