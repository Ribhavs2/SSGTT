from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "./models/meta-llama-3"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,  
    device_map="auto"  
)

input_text = "What are the key challenges of AI in finance?"

inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

output = model.generate(inputs["input_ids"], max_length=200, temperature=0.7)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
