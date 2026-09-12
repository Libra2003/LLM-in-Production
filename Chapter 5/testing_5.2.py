from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_dir = "./models/betterGPT/"

# 1. Load your saved, fine-tuned model and tokenizer
print("Loading saved model...")
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
model = GPT2LMHeadModel.from_pretrained(model_dir)

# 2. Define a prompt string (replacing the undefined 'input')
prompt_text = "Raskolnikov walked into the room and"

# 3. Tokenize correctly
tokenized_input = tokenizer(prompt_text, return_tensors="pt")

# 4. Generate
out = model.generate(
    input_ids=tokenized_input["input_ids"],
    attention_mask=tokenized_input["attention_mask"],
    max_length=256,
    num_beams=5,
    temperature=0.7,
    top_k=50,
    top_p=0.90,
    no_repeat_ngram_size=2,
)

print("\n--- GENERATED TEXT ---")
print(tokenizer.decode(out[0], skip_special_tokens=True))