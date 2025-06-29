from transformers import GPT2LMHeadModel, AutoTokenizer

MODEL_PATH = "C:\Users\Aditya\OneDrive\Documents\Web Scraping\venv\fine_tuned_gpt2"  # Change this to your actual model path

# Load model & tokenizer
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Test if the model can recall something meaningful
test_input = "happy: "
input_ids = tokenizer.encode(test_input, return_tensors="pt")

output = model.generate(input_ids, max_length=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
