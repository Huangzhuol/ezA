import os
import jsonlines
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import numpy as np
import openai
import re
from openai import OpenAI
# ========== CONFIGURATION ==========
openai.api_key = "openai_key" # Replace with your OpenAI API key
Deepseek_key="deepseek_key" # Replace with your DeepSeek key
DATA_DIR = "./acord_data"
TOP_K = 5


# ========== LOAD DATA ==========
def load_jsonl(filepath):
    with jsonlines.open(filepath) as reader:
        return {item['_id']: item['text'] for item in reader}

corpus = load_jsonl(os.path.join(DATA_DIR, "corpus.jsonl"))
queries = load_jsonl(os.path.join(DATA_DIR, "queries.jsonl"))

# ========== EMBEDDINGS ==========
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
clause_ids = list(corpus.keys())
clause_texts = list(corpus.values())
clause_embeddings = model.encode(clause_texts, convert_to_tensor=True)

# ========== GPT-4o UTILS ==========
def generate_explanation_gpt4o(query, clause):
    prompt = f"""
You are a legal contract assistant.

Query: {query}

Clause:
\"\"\"{clause}\"\"\"

Please explain how this clause is relevant to the query.
"""
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Explanation error: {e}")
        return "Explanation failed."

def gpt4omini_relevance_score(query, clause, explanation):
    prompt = f"""
You are a legal contract evaluation assistant.

Query: {query}

Clause:
\"\"\"{clause}\"\"\"

Explanation:
\"\"\"{explanation}\"\"\"

Rate how well this clause and explanation together answer the query, on a scale of 1 (irrelevant) to 5 (highly relevant). Only return a number.
"""
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        content = response.choices[0].message.content.strip()
        match = re.search(r"\b[1-5]\b", content)
        return int(match.group(0)) if match else 1
    except Exception as e:
        print(f"Scoring error: {e}")
        return 1


def deepseek_relevance_score(query, clause, explanation):
    prompt = f"""
You are a legal contract evaluation assistant.

Query: {query}

Clause:
\"\"\"{clause}\"\"\"

Explanation:
\"\"\"{explanation}\"\"\"

Rate how well this clause and explanation together answer the query, on a scale of 1 (irrelevant) to 5 (highly relevant). Only return a number.
"""
    try:
        client = OpenAI(api_key=Deepseek_key, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        content = response.choices[0].message.content.strip()
        match = re.search(r"\b[1-5]\b", content)
        return int(match.group(0)) if match else 1
    except Exception as e:
        print(f"Scoring error: {e}")
        return 1
    
# ========== MAIN LOOP ==========
explanations_gpt = []
explanations_deepseek = []

for qid, query in tqdm(queries.items(), desc="Generating Explanations"):
    q_embedding = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(q_embedding, clause_embeddings, top_k=TOP_K)[0]
    top_clause_ids = [clause_ids[hit['corpus_id']] for hit in hits]

    for cid in top_clause_ids:
        clause = corpus[cid]
        explanation = generate_explanation_gpt4o(query, clause)

        score_gpt = gpt4omini_relevance_score(query, clause, explanation)

        explanations_gpt.append({
            "query_id": qid,
            "clause_id": cid,
            "clause": clause,
            "explanation": explanation,
            "score": score_gpt
        })

        score_deepseek = deepseek_relevance_score(query, clause, explanation)

        explanations_deepseek.append({
            "query_id": qid,
            "clause_id": cid,
            "clause": clause,
            "explanation": explanation,
            "score": score_deepseek
        })

    break #to do one query

# ========== OUTPUT ==========
pd.DataFrame(explanations_gpt).to_csv("acord_explanations_gpt.csv", index=False)
print("Saved explanations to acord_explanations_gpt.csv")

pd.DataFrame(explanations_deepseek).to_csv("acord_explanations_deepseek.csv", index=False)
print("Saved explanations to acord_explanations_deepseek.csv")