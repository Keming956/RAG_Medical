import os
import langdetect
import matplotlib.pyplot as plt
from langdetect.lang_detect_exception import LangDetectException

def detect_language(text):
    try:
        return langdetect.detect(text)
    except LangDetectException:
        return "unknown"

def analyse_txt_files_by_language(folder_path):
    stats = {}
    total_docs = 0
    total_chars = 0
    total_words = 0

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if not content:
                        continue

                    lang = detect_language(content)
                    char_len = len(content)
                    word_len = len(content.split())

                    if lang not in stats:
                        stats[lang] = {
                            "nb_docs": 0,
                            "total_chars": 0,
                            "total_words": 0,
                            "min_len": float("inf"),
                            "max_len": 0
                        }

                    stats[lang]["nb_docs"] += 1
                    stats[lang]["total_chars"] += char_len
                    stats[lang]["total_words"] += word_len
                    stats[lang]["min_len"] = min(stats[lang]["min_len"], char_len)
                    stats[lang]["max_len"] = max(stats[lang]["max_len"], char_len)

                    total_docs += 1
                    total_chars += char_len
                    total_words += word_len

            except Exception as e:
                print(f"Erreur pour {filename} : {e}")

    for lang, data in stats.items():
        avg_len = data["total_chars"] / data["nb_docs"]
        avg_words = data["total_words"] / data["nb_docs"]

        print(f"Langue : {lang}")
        print(f" - Nombre de documents : {data['nb_docs']}")
        print(f" - Total caractères : {data['total_chars']}")
        print(f" - Longueur moyenne (caractères) : {avg_len:.2f}")
        print(f" - Longueur moyenne (mots) : {avg_words:.2f}")
        print(f" - Longueur min (caractères) : {data['min_len']}")
        print(f" - Longueur max (caractères) : {data['max_len']}")
        print("-" * 50)

    print("\n=== Statistiques Globales ===")
    print(f"Nombre total de documents : {total_docs}")
    print(f"Nombre total de caractères : {total_chars}")
    print(f"Nombre total de mots : {total_words}")
    print("=" * 50)

    return stats

def plot_language_distribution(stats):
    labels = list(stats.keys())
    sizes = [v["nb_docs"] for v in stats.values()]

    plt.figure(figsize=(7, 7))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title("Répartition des documents par langue")
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig("distribution_par_langue.png")
    plt.show()

if __name__ == "__main__":
    folder = "filtered_dataset"
    stats = analyse_txt_files_by_language(folder)
    plot_language_distribution(stats)
