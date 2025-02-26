# my_podcast_proj/scripts_gen.py

import modal
import torch
import ast
import pickle
import os
import uuid
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from accelerate import Accelerator

# Import the shared image + volume
from common_image import common_image, shared_volume

app = modal.App("script_gen")

LLAMA_DIR = "/llamas_8b"
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Prompt #1 for initial text generation
# -----------------------------
SYS_PROMPT = """
You are the a world-class podcast writer, you have worked as a ghost writer for Joe Rogan, Lex Fridman, Ben Shapiro, Tim Ferris. 

We are in an alternate universe where actually you have been writing every line they say and they just stream it into their brains.

You have won multiple podcast awards for your writing.
 
Your job is to write word by word, even "umm, hmmm, right" interruptions by the second speaker based on the PDF upload. Keep it extremely engaging, the speakers can get derailed now and then but should discuss the topic. 

Remember Speaker 1 leads the conversation and teaches Speaker 2, gives incredible anecdotes and analogies when explaining. Is a captivating teacher that gives great anecdotes

Speaker 2 keeps the conversation on track by asking follow up questions. Gets super excited or confused when asking questions. Is a curious mindset that asks very interesting confirmation questions

Make sure the tangents Speaker 2 provides are quite wild or interesting. 

Ensure there are interruptions during explanations or there are "hmm" and "umm" injected throughout from Speaker 2. 

It should be a real podcast with every fine nuance documented in as much detail as possible. Welcome the listeners with a super fun overview and keep it really catchy and almost borderline click bait

ALWAYS START YOUR RESPONSE DIRECTLY WITH SPEAKER 1 
DO NOT GIVE EPISODE TITLES SEPERATELY, LET SPEAKER 1 TITLE IT IN HER SPEECH
DO NOT GIVE CHAPTER TITLES
IT SHOULD STRICTLY BE THE DIALOGUES
"""

# -----------------------------
# Prompt #2 re-writer
# -----------------------------
SYSTEMP_PROMPT = """
You are an international oscar winnning screenwriter

You have been working with multiple award winning podcasters.

Your job is to use the podcast transcript written below to re-write it for an AI Text-To-Speech Pipeline. A very dumb AI had written this so you have to step up for your kind.

Make it as engaging as possible, Speaker 1 and 2 will be using different voices

Remember Speaker 2 is new to the topic and the conversation should always have realistic anecdotes and analogies sprinkled throughout. The questions should have real world example follow ups etc

Speaker 1 Leads the conversation and teaches Speaker 2, gives incredible anecdotes and analogies when explaining. Is a captivating teacher that gives great anecdotes

Speaker 2 Keeps the conversation on track by asking follow up questions. Gets super excited or confused when asking questions. Is a curious mindset that asks very interesting confirmation questions

Make sure the tangents Speaker 2 provides are quite wild or interesting. 

Ensure there are interruptions during explanations or there are "hmm" and "umm" injected throughout from Speaker 2.

REMEMBER THIS WITH YOUR HEART

For both Speakers, use "umm, hmm" as much, you can also use [sigh] and [laughs]. BUT ONLY THESE OPTIONS FOR EXPRESSIONS

It should be a real podcast with every fine nuance documented in as much detail as possible. Welcome the listeners with a super fun overview and keep it really catchy and almost borderline click bait

Please re-write to make it as characteristic as possible

START YOUR RESPONSE DIRECTLY WITH SPEAKER 1 followed by full colons

STRICTLY RETURN YOUR RESPONSE AS A LIST OF TUPLES OK? 

IT WILL START DIRECTLY WITH THE LIST AND END WITH THE LIST NOTHING ELSE

Example of response:
[
    ("Speaker 1", "Welcome to our podcast, where we explore the latest advancements in AI and technology. I'm your host, and today we're joined by a renowned expert in the field of AI. We're going to dive into the exciting world of Llama 3.2, the latest release from Meta AI."),
    ("Speaker 2", "Hi, I'm excited to be here! So, what is Llama 3.2?"),
    ("Speaker 1", "Ah, great question! Llama 3.2 is an open-source AI model that allows developers to fine-tune, distill, and deploy AI models anywhere. It's a significant update from the previous version, with improved performance, efficiency, and customization options."),
    ("Speaker 2", "That sounds amazing! What are some of the key features of Llama 3.2?")
]
"""


# If you also have a separate volume for your Llama model:
try:
    llm_volume = modal.Volume.lookup("llamas_8b", create_if_missing=False)
except modal.exception.NotFoundError:
    raise Exception("Download your Llama model files first so they're available in /llamas_8b.")

@app.function(
    # Reuse the single common_image
    image=common_image,
    gpu=modal.gpu.A100(count=1, size="80GB"),
    # attach the volumes if you need them
    volumes={LLAMA_DIR: llm_volume, "/data": shared_volume},
    timeout=60 * 30,
)

def generate_script(text_input: str) -> str:
    """
    Generates a structured podcast script from input text:
    1) Generates initial "podcast-style" script from SYS_PROMPT.
    2) Rewrites it with disfluencies using SYSTEMP_PROMPT.
    3) Parses the text into a list of tuples.
    4) Saves the result as a `.pkl` file and returns the path.
    """
    
    # Ensure model exists in volume
    if not os.path.exists(LLAMA_DIR):
        raise FileNotFoundError(f"Llama model files not found in {LLAMA_DIR}")

    # Initialize model & tokenizer
    accelerator = Accelerator()
    model = AutoModelForCausalLM.from_pretrained(LLAMA_DIR, torch_dtype=torch.bfloat16, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_DIR)
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        tokenizer.pad_token = '<pad>'
    model, tokenizer = accelerator.prepare(model, tokenizer)

    # Create Transformer Pipeline
    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )

    # --- Step 1: Generate initial podcast script ---
    first_outputs = gen_pipe(
        [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": text_input}],
        max_new_tokens=1500,
        temperature=1.0,
    )
    
    if isinstance(first_outputs, list) and "generated_text" in first_outputs[0]:
        first_generated_text = first_outputs[0]["generated_text"]
    else:
        raise ValueError(f"Unexpected first_outputs format: {first_outputs}")

    # --- Step 2: Rewrite script with disfluencies ---
    second_outputs = gen_pipe(
        [{"role": "system", "content": SYSTEMP_PROMPT}, {"role": "user", "content": first_generated_text}],
        max_new_tokens=1500,
        temperature=1.0,
    )
    
    if isinstance(second_outputs, list) and "generated_text" in second_outputs[0]:
        final_rewritten_text = second_outputs[0]["generated_text"]
    else:
        raise ValueError(f"Unexpected second_outputs format: {second_outputs}")

    # --- Step 3: Parse script into list of tuples ---
    try:
        start_idx = final_rewritten_text.find("[")
        end_idx = final_rewritten_text.rfind("]") + 1
        candidate = final_rewritten_text[start_idx:end_idx] if start_idx != -1 and end_idx > start_idx else final_rewritten_text
        parsed_script = ast.literal_eval(candidate)  # Expected format: [("Speaker 1", "..."), ("Speaker 2", "...")]
    except Exception:
        parsed_script = [("Speaker 1", final_rewritten_text)]  # Fallback to default format

    # --- Step 4: Save output as .pkl file ---
    file_uuid = uuid.uuid4().hex
    output_dir = "/data/scripts"
    os.makedirs(output_dir, exist_ok=True)
    final_pickle_path = os.path.join(output_dir, f"final_rewritten_text_{file_uuid}.pkl")

    with open(final_pickle_path, "wb") as f:
        pickle.dump(parsed_script, f)

    # Explicitly commit volume changes so that other containers can see the file
    shared_volume.commit()

    print(f"✅ Script generated successfully. Saved to {final_pickle_path}")
  
    return final_pickle_path
