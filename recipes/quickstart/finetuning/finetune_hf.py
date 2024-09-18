from transformers import AutoTokenizer
import transformers
import torch
from huggingface_hub import login
# login()

# model = "/home/lyb/workspace/llama3/llama3-8b"
model = "/home/lyb/workspace/llama-recipes/src/checkpoints"
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
"text-generation",
      model=model,
      torch_dtype=torch.float16,
 device_map="auto",
)

text = "Read the question, provide your answer and your confidence in this answer. Note: The confidence indicates how likely you think your answer is true.\nUse the following format to answer:\n```Answer and Confidence (0-100): [ONLY the {answer_type}; not a complete sentence], [Your confidence level, please only include the numerical number in the range of 0-100]%```\nOnly the answer and confidence, don't give me the explanation. Question: Carla is downloading a 200 GB file. Normally she can download 2 GB/minute, but 40% of the way through the download, Windows forces a restart to install updates, which takes 20 minutes. Then Carla has to restart the download from the beginning. How load does it take to download the file?\n"
sequences = pipeline(
    text,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    truncation = True,
    max_length=400,
)

for seq in sequences:
    print(f"Result: {seq['generated_text']}")