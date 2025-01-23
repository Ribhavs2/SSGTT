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

def generate_paraphrases_from_file(pipeline, input_file: str, output_file: str, batch_size: int = 5, max_new_tokens: int = 200, temperature: float = 0.4, num_repeats: int = 2):
    """
    Read input texts from a file, generate paraphrases using batch inference, and save results to another file.
    
    :param pipeline: The loaded text-generation pipeline.
    :param input_file: Path to the input text file (each line is a text input).
    :param output_file: Path to the output file where paraphrased texts will be saved.
    :param batch_size: Number of paraphrases to generate per batch.
    :param max_new_tokens: Maximum number of new tokens to generate.
    :param temperature: Sampling temperature.
    :param num_repeats: Number of times to paraphrase each input.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        input_texts = [line.strip() for line in f.readlines()]

    paraphrases = []
    num_batches = (len(input_texts) + batch_size - 1) // batch_size  # 计算 batch 数

    for batch_idx in range(num_batches):
        batch_inputs = input_texts[batch_idx * batch_size: (batch_idx + 1) * batch_size]

        for repeat in range(num_repeats):  # 每个输入推理 num_repeats 次
            batch_results = generate_responses(
                pipeline,
                "You are a helpful assistant who paraphrases sentences.",
                [f"Paraphrase this sentence: {text}" for text in batch_inputs],
                max_new_tokens,
                temperature
            )

            for i, (para, response_time) in enumerate(batch_results):
                input_text = batch_inputs[i]
                paraphrases.append(f"Input: {input_text}\nParaphrase {repeat+1}: {para}\nTime: {response_time:.4f} sec\n")
                print(f"Batch {batch_idx+1}/{num_batches}, Repeat {repeat+1}: {input_text} → {para} ({response_time:.4f}s)")

    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines("\n".join(paraphrases))

    print(f"\nAll paraphrases saved to {output_file}")

# Example usage
if __name__ == "__main__":
    model_id = "./models/Llama-3.1-8B-Instruct"
    pipeline = load_pipeline(model_id)

    input_file = "output_texts.txt"   # 100 line input_text
    output_file = "paraphrased_outputs.txt"

    generate_paraphrases_from_file(pipeline, input_file, output_file, batch_size=5, num_repeats=2)
