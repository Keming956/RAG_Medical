import os
import json
from transformers import pipeline
from tqdm import tqdm

# ========== 配置 ==========
input_folder = "./dataset"                     # 文本所在文件夹
output_folder = "./filtered_dataset"          # 输出文件夹
max_chars = 1000                               # 每篇只取前 1000 字判断（可调）
model_name = "google/flan-t5-base"             # 使用轻量模型
device = -1                                    # -1 表示使用 CPU，若有 GPU 改为 0

# ========== 加载模型 ==========
print("正在加载模型...")
classifier = pipeline("text2text-generation", model=model_name, device=device)
print("模型加载完成。")

# ========== 判断函数 ==========
def is_psych_related(text: str) -> bool:
    prompt = (
        "Please determine whether the following English text is primarily about psychological disorders, "
        "such as depression, anxiety, schizophrenia, suicide, autism, eating disorders, or trauma.\n\n"
        "Answer 'yes' or 'no'.\n\n"
        f"Text:\n{text[:max_chars]}"
    )
    response = classifier(prompt, max_new_tokens=10)[0]["generated_text"].lower()
    return "yes" in response

# ========== 筛选文本 ==========
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

kept = 0
total = 0

print(f"开始筛选 {input_folder} 中的 .txt 文件...")

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

print(f"\n✅ 筛选完成，共处理 {total} 篇，保留 {kept} 篇心理疾病相关文本。")
print(f"结果保存在：{output_folder}")
