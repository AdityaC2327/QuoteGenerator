import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import AutoTokenizer  # For flexibility

# Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Ensure tokenizer handles padding
tokenizer.pad_token = tokenizer.eos_token

# Load dataset (quotes categorized by mood)
df = pd.read_csv("quotes_by_mood.csv")

# Format the data for training
df['input_text'] = df['mood'] + ": " + df['quote']

# Split the dataset into training and validation sets BEFORE converting to Dataset format
train_texts, val_texts = train_test_split(df[['input_text']], test_size=0.2, random_state=42)

# Function to tokenize the data
def tokenize_function(examples):
    return tokenizer(examples['input_text'], truncation=True, padding="max_length", max_length=128)

# Convert to Hugging Face Dataset format and tokenize in one step
train_dataset = Dataset.from_pandas(train_texts).map(tokenize_function, batched=True)
val_dataset = Dataset.from_pandas(val_texts).map(tokenize_function, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # Corrected to eval_strategy
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,  # Add eval batch size for validation
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=10_000,
    save_total_limit=2,
)

# Define a data collator for padding dynamically during training
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # We are training causal LM (GPT-2), not masked LM
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,  # Use tokenizer here
    data_collator=data_collator
)

# Fine-tune the model
trainer.train()

# Save the model after training
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# Function to generate a quote based on mood
def generate_quote(mood):
    input_text = mood + ": "
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Create an attention mask where 1 means the token is part of the input
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

    # Generate text
    output = model.generate(
        input_ids, 
        attention_mask=attention_mask,  # Pass attention mask
        max_length=50, 
        num_return_sequences=1, 
        no_repeat_ngram_size=2, 
        temperature=0.7, 
        do_sample=True  # Ensures sampling with temperature
    )

    # Decode and return the generated text
    quote = tokenizer.decode(output[0], skip_special_tokens=True)
    return quote

# Example: Generate a quote for a "happy" mood
generated_quote = generate_quote("happy")
print(generated_quote)
