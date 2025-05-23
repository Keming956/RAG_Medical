from transformers import pipeline
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import AnswerRelevancyMetric, ContextualPrecisionMetric
from deepeval.evaluate import evaluate
from deepeval.metrics.base_metric import BaseMetric
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Optional, List

from rag import load_documents, split_documents, create_vectorstore, load_cpu_friendly_llm, create_qa_chain

document_directory = "./dataset/"
documents = load_documents(document_directory)
chunks = split_documents(documents)
vectorstore, _ = create_vectorstore(chunks)
llm = load_cpu_friendly_llm()
qa_chain = create_qa_chain(vectorstore, llm, use_reranking=True)

generator = pipeline("text2text-generation", model="google/flan-t5-small", device=-1)

def generate_synthetic_qa(context):
    q_prompt = (
    f"Based on the following text about mental health or psychological disorders, "
    f"generate a clinically relevant question that could be answered by the text:\n{context}"
)

    a_prompt = (
    f"The following text discusses topics related to mental health or psychological conditions. "
    f"Based on it, answer the question below in a medically accurate and concise way:\n{context}"
)


    question = generator(q_prompt, max_new_tokens=50)[0]["generated_text"]
    answer = generator(f"{a_prompt}\nQuestion: {question}", max_new_tokens=80)[0]["generated_text"]

    return question.strip(), answer.strip()

samples = []
for doc in chunks[:5]: 
    context_text = doc.page_content
    question, reference_answer = generate_synthetic_qa(context_text)
    rag_result = qa_chain({"query": question})["result"]

    samples.append(LLMTestCase(
        input=question,
        context=[context_text],
        expected_output=reference_answer,
        actual_output=rag_result
    ))

synthetic_dataset = EvaluationDataset(test_cases=samples)

class EmbeddingSimilarityMetric(BaseMetric):
    def __init__(self, threshold=0.75,
                 model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.threshold = threshold
        self.model = SentenceTransformer(model_name)

    def measure(self, test_case: LLMTestCase):

        context = " ".join(test_case.context)
        emb = self.model.encode([test_case.actual_output, context],
                                normalize_embeddings=True)
        score = float(cosine_similarity(emb[:1], emb[1:])[0][0])
        #test_case.metric_scores[self.name()] = score
        return score

    def is_pass(self, score: float) -> bool:
        return score >= self.threshold

    def name(self):
        return "embedding_similarity"

    def rationale(self) -> str:
        return ("Cosine similarity between generated answer and its "
                "supporting context")

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)


metric = EmbeddingSimilarityMetric(threshold=0.75)
results = []
for ex in samples:
    results.append(metric.measure(ex))


print("Question:", samples[0].input)
print("Expected Answer:", samples[0].expected_output)
print("RAG Answer:", samples[0].actual_output)
print("Similarity Score:", results)

import csv

csv_file = "evaluation_results.csv"

with open(csv_file, mode="w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Sample Index", "Question", "Expected Answer", "RAG Answer", "Similarity Score"])

    for i, sample in enumerate(samples):
        writer.writerow([
            i + 1,
            sample.input,
            sample.expected_output,
            sample.actual_output,
            round(results[i], 4)
        ])

print(f"Results saved as CSV file：{csv_file}")
