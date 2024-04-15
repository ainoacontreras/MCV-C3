from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import tqdm
from diffusers import AutoPipelineForText2Image
import matplotlib.pyplot as plt
import time

@torch.no_grad()
def generate_similar_prompt(model, tokenizer, captions, creativity=0.6):
    """
    Generates similar but uniquely different prompts for a batch of original captions using a chat-based template.

    Parameters:
    - model: The preloaded generative AI model.
    - tokenizer: The tokenizer corresponding to the model.
    - captions: A list of original captions to generate variations for.
    - creativity: A float indicating how creative the variations should be (affects sampling).

    Returns:
    A list of generated variations.
    """
    # compute the time of execution
    start = time.time()

    chat = [f"Instruct: Generate one variation of the description that could lead to similar but distinct image (the new description must ommit some elements from the original or add new ones). The image must be plausible in the real world, describing real scenarios. Do not include the original prompt. Here's the description: {caption}\n Output:" for caption in captions]

    # Prepare the input for the model
    input_ids = tokenizer(chat, return_tensors="pt", return_attention_mask=False, padding=True).to("cuda")
    
    # Generate responses
    outputs = model.generate(**input_ids, max_new_tokens=30, temperature=creativity, do_sample=True, pad_token_id=tokenizer.eos_token_id)

    # Decode the generated prompts
    generated_prompt = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # compute the time of execution
    end = time.time()
    print(f"Time taken to generate prompts: {end-start} seconds")

    return [c.split("Output: ")[-1].rstrip('\n') for c in generated_prompt]

@torch.no_grad()
def generate_images(pipe, caption, plot_name=None):
    """
    Generates images based on a list of captions using a text-to-image pipeline.

    Parameters:
    - pipe: The text-to-image pipeline.
    - captions: A caption to generate an image for.

    Returns:
    A generated image corresponding to the input caption.
    """
    prompt = "A high-resolution, professional, highly detailed, realistic photograph of " + caption
    negative_prompt = "anime, cartoon, graphic, painting, crayon, graphite, abstract glitch, blurry"
    image = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=1.5, negative_prompt=negative_prompt).images[0].resize((224, 224))
    # print(image.size)

    if plot_name is not None:
        #save each image
        image.save(f"{plot_name}")

    return image


if __name__ == "__main__":
    
    # Initialize the model and tokenizer
    model_id = "microsoft/phi-2"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16", device_map="auto")

    captions = ['A large brick building with a tall tower containing a clock near the top',
    'A woman riding on a duck boat on the lake',
    'A man riding a motorcycle across a lush green park.',
    'A few plates with various entrees on them.',
    'A brown horse standing in a lush green field.',
    'A firetruck with lights on is on a city street.',
    'A man is searching for food in a refrigerator.',
    'Two motorcycles and a bicycle parked against a sliding door. ',
    'A table filled with a cake and paper plates with ice cream and cake.',
    'The man is playing tennis on the tennis court. ',
    'A man in striped sweater putting a whole donut in his mouth.',
    'Several empty boats floating on the river on a cloudy day.',
    'A group children pose together for a picture.',
    'A bicycle is parked next to a small desk.',
    'A woman holding a red umbrella sits on a bench facing the sea.']

    # Assuming the model and tokenizer are already initialized and properly configured
    similar_prompts = generate_similar_prompt(model, tokenizer, captions)

    with open("generated_images/generated_prompts.txt", "w") as f:
        for text in similar_prompts:
            f.write(f"{text}\n")

    # Print or process the generated prompts as needed
    for i,text in enumerate(similar_prompts):
        generate_images(pipe, text, plot_name=f"generated_images/image_{i}.png")
