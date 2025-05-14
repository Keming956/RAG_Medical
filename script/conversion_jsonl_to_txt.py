import json
import os


input_path = "combined.jsonl"

output_dir = "corpus"


os.makedirs(output_dir, exist_ok=True)

with open(input_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        data = json.loads(line)

        file_title = data.get("title", f"doc_{i}")
        file_title = "".join(c for c in file_title if c.isalnum() or c in (" ", "_", "-")).rstrip()
        text = data.get("text", "")


        output_path = os.path.join(output_dir, f"{file_title}.txt")
        with open(output_path, "w", encoding="utf-8") as out_file:
            out_file.write(text)

print("finished")