import os
import pandas as pd
import torch
from datasets import Dataset, load_from_disk
from sklearn.model_selection import train_test_split
from transformers import (GPT2LMHeadModel, AutoTokenizer, Trainer, 
                          TrainingArguments, DataCollatorForLanguageModeling)

# ========================
# 1. Data Preprocessing
# ========================

def clean_text(text):
    text = text.lower().replace("\n", " ").strip()
    return text

# Load dataset
df = pd.read_csv("quotes_by_mood.csv")

def transform_mood(mood):
    mood_mapping = {
        "angry": "calm", 
        "stressed": "motivated", 
        "sad": "hopeful",
        "frustrated": "relaxed",
        "lonely": "loved",
        "anxious": "peaceful",
        "happy": "joyful",
        "bliss": "elevated",
        "motivated": "unstoppable",
        "relaxed": "peaceful"
    }
    return mood_mapping.get(mood.lower(), mood)  # Transform negative moods

df["Cleaned_Quote"] = df["quote"].apply(clean_text)
df["Transformed_Mood"] = df["mood"].apply(transform_mood)
df["input_text"] = df["Transformed_Mood"] + ": " + df["Cleaned_Quote"]

# Train-validation split
train_texts, val_texts = train_test_split(df[["input_text"]], test_size=0.2, random_state=42)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["input_text"], truncation=True, padding="max_length", max_length=128)

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_texts).map(tokenize_function, batched=True)
val_dataset = Dataset.from_pandas(val_texts).map(tokenize_function, batched=True)

# Save datasets
train_dataset.save_to_disk("train_dataset")
val_dataset.save_to_disk("val_dataset")
print("Data preprocessing complete.")

# ========================
# 2. Model Training (Skip if already trained)
# ========================

model_path = "./fine_tuned_gpt2"
if os.path.exists(model_path):
    print("Pre-trained model found. Skipping training.")
    model = GPT2LMHeadModel.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
else:
    # Load datasets
    train_dataset = load_from_disk("train_dataset")
    val_dataset = load_from_disk("val_dataset")

    # Load GPT-2 model
    model = GPT2LMHeadModel.from_pretrained("gpt2").to("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer.pad_token = tokenizer.eos_token

    # Training setup
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=model_path,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        push_to_hub=False,
        fp16=torch.cuda.is_available(),
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print("Model training complete. Saved fine-tuned model.")

# ========================
# 3. Quote Generation (With Mood Transformation)
# ========================

def generate_quote(mood):
    transformed_mood = transform_mood(mood)  # Transform negative moods to positive ones
    input_text = f"{transformed_mood}: "
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
    attention_mask = torch.ones_like(input_ids).to(model.device)  # Fixing attention mask issue
    
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=108, 
        num_return_sequences=1,
        do_sample=True,
        no_repeat_ngram_size=2,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        num_beams=5,  # Beam search for better output
        pad_token_id=tokenizer.eos_token_id
    )
    
    return tokenizer.decode(output[0], skip_special_tokens=True).replace(input_text, "").strip()

if __name__ == "__main__":
    mood = input("Enter a mood: ")
    print(f"Generated {mood} quote: \"{generate_quote(mood)}\"")
