import transformers
import torch
import time

def load_pipeline(model_id: str, device: int = 0):
    """
    Load the text-generation pipeline with the specified model.
    
    :param model_id: Path to the model directory or model identifier.
    :param device: Device to load the model on (0 for GPU, -1 for CPU).
    :return: A text-generation pipeline.
    """
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    pipeline.tokenizer.padding_side = "left" 
    if pipeline.tokenizer.pad_token is None:
        pipeline.tokenizer.pad_token = pipeline.tokenizer.eos_token 

    return pipeline

def generate_responses(pipeline, system_prompt: str, user_inputs: list, max_new_tokens: int = 256, temperature: float = 0.4):
    """
    Generate responses for a batch of inputs using Llama 3 chat template.
    
    :param pipeline: The loaded text-generation pipeline.
    :param system_prompt: The system role prompt.
    :param user_inputs: A list of user input messages.
    :param max_new_tokens: Maximum number of new tokens to generate.
    :param temperature: Sampling temperature.
    :return: A list of tuples containing the model's response texts and response times.
    """
    messages_list = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
        for user_input in user_inputs
    ]

    # **Use Llama 3 chat template**
    prompts = [pipeline.tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages_list]

    # terminators`<|eot_id|>` or `eos_token_id`
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    start_time = time.time()
    outputs = pipeline(
        prompts,
        max_new_tokens=max_new_tokens,
        eos_token_id=terminators,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        batch_size=len(user_inputs)
    )
    end_time = time.time()

    response_time = end_time - start_time

    cleaned_outputs = []
    for output in outputs:
        response_text = output[0]["generated_text"]
        
        response_text = response_text.split("<|eot_id|>")[-1]  
        response_text = response_text.replace("<|start_header_id|>assistant<|end_header_id|>", "").strip()
        
        cleaned_outputs.append((response_text, response_time))

    return cleaned_outputs

def generate_paraphrases(pipeline, input_texts: list, batch_size: int = 5, max_new_tokens: int = 200, temperature: float = 0.4, num_repeats: int = 2):
    """
    Generate paraphrases for a list of input texts using batch inference.
    
    :param pipeline: The loaded text-generation pipeline.
    :param input_texts: A list of input texts to be paraphrased.
    :param batch_size: Number of paraphrases to generate per batch.
    :param max_new_tokens: Maximum number of new tokens to generate.
    :param temperature: Sampling temperature.
    :param num_repeats: Number of times to paraphrase each input.
    :return: A list of paraphrased texts.
    """
    paraphrases = []
    num_batches = (len(input_texts) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        batch_inputs = input_texts[batch_idx * batch_size: (batch_idx + 1) * batch_size]

        for _ in range(num_repeats):
            batch_results = generate_responses(
                pipeline,
                "You are a helpful assistant who paraphrases sentences.",
                [f"Paraphrase this sentence: {text}" for text in batch_inputs],
                max_new_tokens,
                temperature
            )

            paraphrases.extend([para for para, _ in batch_results])

    return paraphrases

# Example usage
if __name__ == "__main__":
    model_id = "./models/Llama-3.1-8B-Instruct"
    pipeline = load_pipeline(model_id)

    input_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Learning new languages can be a rewarding experience."
    ]

    paraphrased_texts = generate_paraphrases(pipeline, input_texts, batch_size=3, num_repeats=2)
    print(paraphrased_texts)

    for idx, para in enumerate(paraphrased_texts):
        print(f"Paraphrase {idx+1}: {para}")
