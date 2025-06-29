import os
import torch
from transformers import GPT2LMHeadModel, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_from_disk

# Load preprocessed datasets
train_dataset = load_from_disk(r"C:\Users\Aditya\OneDrive\Documents\Web Scraping\venv\train_dataset")
val_dataset = load_from_disk(r"C:\Users\Aditya\OneDrive\Documents\Web Scraping\venv\val_dataset")

# Load GPT-2 model & tokenizer
model_name = "gpt2"
device = "cuda" if torch.cuda.is_available() else "cpu"  # Check for GPU availability
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set padding token to avoid warnings
tokenizer.pad_token = tokenizer.eos_token  
model.config.pad_token_id = tokenizer.pad_token_id  # Ensure model uses pad token

# Data collator for dynamic padding
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # GPT-2 is a causal LM, not masked LM
)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./trained_model",
    evaluation_strategy="epoch",  # Validate after every epoch
    eval_steps=500,  # If dataset is small, change to "steps" and use this
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",  # Save model every epoch
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
    push_to_hub=False,  # Prevents accidental uploads to Hugging Face
    fp16=True if torch.cuda.is_available() else False,  # Enable mixed precision training if on GPU
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Train the model
trainer.train()

# Ensure the save directory exists
save_dir = "./fine_tuned_gpt2"
os.makedirs(save_dir, exist_ok=True)

# Save the fine-tuned model and tokenizer
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print(f"Training complete! Model saved to '{save_dir}'.")
