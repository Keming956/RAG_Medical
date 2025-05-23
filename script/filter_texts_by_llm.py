import os
import json
from transformers import pipeline
from tqdm import tqdm

input_folder = "./dataset"
output_folder = "./filtered_dataset"
max_chars = 1000
model_name = "google/flan-t5-base"
device = -1

print("Loading model...")
classifier = pipeline("text2text-generation", model=model_name, device=device)
print("Model loaded")

def is_psych_related(text: str) -> bool:
    prompt = (
        "Please determine whether the following English text is primarily about psychological disorders, "
        "such as depression, anxiety, schizophrenia, suicide, autism, eating disorders, or trauma.\n\n"
        "Answer 'yes' or 'no'.\n\n"
        f"Text:\n{text[:max_chars]}"
    )
    response = classifier(prompt, max_new_tokens=10)[0]["generated_text"].lower()
    return "yes" in response

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

kept = 0
total = 0

print(f"Start filtering .txt files in {input_folder}...")

for fname in tqdm(os.listdir(input_folder)):
    if not fname.endswith(".txt"):
        continue
    total += 1

    full_path = os.path.join(input_folder, fname)
    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    if is_psych_related(text):
        out_path = os.path.join(output_folder, fname)
        with open(out_path, "w", encoding="utf-8") as fout:
            fout.write(text)
        kept += 1

print(f"\n Filtering done，have processed {total} files，kept {kept} files related to psychological disorders")
print(f"Resulats saved as：{output_folder}")
