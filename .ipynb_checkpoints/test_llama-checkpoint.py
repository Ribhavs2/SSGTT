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
    return transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device=device,
    )

def generate_response(pipeline, system_prompt: str, user_input: str, max_new_tokens: int = 256, temperature: float = 0.4):
    """
    Generate a response from the model using a chat format.
    
    :param pipeline: The loaded text-generation pipeline.
    :param system_prompt: The system role prompt.
    :param user_input: The user's input message.
    :param max_new_tokens: Maximum number of new tokens to generate.
    :param temperature: Sampling temperature.
    :return: The model's response in text.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    
    start_time = time.time()
    outputs = pipeline(messages, max_new_tokens=max_new_tokens, temperature=temperature)
    end_time = time.time()
    
    response_time = end_time - start_time
    return outputs[0]["generated_text"][-1], response_time

def generate_paraphrases(pipeline, text: str, n: int, max_new_tokens: int = 200, temperature: float = 0.4):
    """
    Generate multiple paraphrases for a given text and measure the time taken for each paraphrase.
    
    :param pipeline: The loaded text-generation pipeline.
    :param text: The input text to paraphrase.
    :param n: Number of paraphrases to generate.
    :param max_new_tokens: Maximum number of new tokens to generate.
    :param temperature: Sampling temperature.
    :return: A list of tuples containing the paraphrased text and the time taken for each paraphrase.
    """
    paraphrases = []
    for i in range(n):
        paraphrase, response_time = generate_response(
            pipeline,
            "You are a helpful assistant who paraphrases sentences.",
            f"Paraphrase this sentence: {text}",
            max_new_tokens,
            temperature
        )
        paraphrases.append((paraphrase["content"], response_time))
        print(f"Paraphrase {i+1} generated in {response_time:.4f} seconds.")
    
    return paraphrases

# Example usage
if __name__ == "__main__":
    model_id = "./models/Llama-3.1-8B-Instruct"
    pipeline = load_pipeline(model_id)
    
    input_text = "Jack the Dog is a 2001 American comedy-drama film, written and directed by Bobby Roth and starring Néstor Carbonell, Barbara Williams, Barry Newman, and Anthony LaPaglia."
    paraphrased_versions = generate_paraphrases(pipeline, input_text, 8)
    
    for i, (para, time_taken) in enumerate(paraphrased_versions, 1):
        print(f"Paraphrase {i}: {para}")
        print(f"Time taken: {time_taken:.4f} seconds\n")
