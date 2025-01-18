from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_model(model_path="./models/meta-llama-3"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_8bit=True,  
        device_map="auto"
    )
    return tokenizer, model

def generate_response(input_text, model, tokenizer, max_length=200, temperature=0.4):
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    output = model.generate(inputs["input_ids"], max_length=max_length, temperature=temperature)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def generate_paraphrases(text, n, model, tokenizer, max_length=200, temperature=0.4):
    paraphrases = []
    for _ in range(n):
        prompt = f"Please paraphrase the following sentence in a different way:\n\n\"{text}\""
        paraphrase = generate_response(prompt, model, tokenizer, max_length, temperature)
        paraphrases.append(paraphrase)
    return paraphrases

if __name__ == "__main__":
    model_path = "./models/meta-llama-3"
    tokenizer, model = load_model(model_path)

    input_text = "Jack the Dog is a 2001 American comedy-drama film, written and directed by Bobby Roth and starring Néstor Carbonell, Barbara Williams, Barry Newman, and Anthony LaPaglia."
    paraphrased_versions = generate_paraphrases(input_text, 3, model, tokenizer)

    for i, para in enumerate(paraphrased_versions, 1):
        print(f"Paraphrase {i}: {para}")
