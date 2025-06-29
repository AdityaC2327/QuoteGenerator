import torch
from transformers import GPT2LMHeadModel, AutoTokenizer

MODEL_PATH = "gpt2"

# Load model & tokenizer
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# Set pad token to avoid warning
tokenizer.pad_token = tokenizer.eos_token

def generate_quote(mood):
    input_text = f"Generate a {mood} quote: "
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    
    # Create attention mask
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
    
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,  # Added attention mask
        max_length=100,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Clean up the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # Remove the input prompt from the result
    return generated_text.replace(input_text, "").strip('"')

if __name__ == "__main__":
    mood = input("Enter a mood: ")
    print(f"Generated {mood} quote: \"{generate_quote(mood)}\"")