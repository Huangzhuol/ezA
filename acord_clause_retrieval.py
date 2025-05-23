
import os
import jsonlines
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import numpy as np
import openai

# ========== CONFIGURATION ==========
openai.api_key = "your-openai-api-key"
DATA_DIR = "./acord_data"
TOP_K = 5  # Number of top clauses to rerank
USE_BM25 = False  # Set to True if you want lexical retrieval as well

# ========== LOAD DATA ==========
def load_jsonl(filepath):
    with jsonlines.open(filepath) as reader:
        return {item['id']: item['text'] for item in reader}

corpus = load_jsonl(os.path.join(DATA_DIR, "corpus.jsonl"))
queries = load_jsonl(os.path.join(DATA_DIR, "queries.jsonl"))
qrels_df = pd.read_csv(os.path.join(DATA_DIR, "qrels.tsv"), sep='\t', names=["query_id", "corpus_id", "score"])

# ========== EMBEDDINGS ==========
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
clause_ids = list(corpus.keys())
clause_texts = list(corpus.values())
clause_embeddings = model.encode(clause_texts, convert_to_tensor=True)

id_to_index = {cid: idx for idx, cid in enumerate(clause_ids)}

# ========== RERANKER ==========
def gpt4o_score(query, clause):
    prompt = f"Query: {query}\nClause: {clause}\nRate the relevance from 1 to 5:"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a legal contract analysis assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        return int(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"Error scoring with GPT-4o: {e}")
        return 1  # fallback score

# ========== METRICS ==========
def precision_at_k(preds, truth, k, star_thresh):
    relevant = set(cid for cid, score in truth if score >= star_thresh)
    top_k = preds[:k]
    return sum(1 for cid in top_k if cid in relevant) / k

def ndcg_at_k(preds, truth, k):
    id_to_relevance = {cid: score for cid, score in truth}
    dcg = sum((2**id_to_relevance.get(cid, 0) - 1) / np.log2(idx + 2) for idx, cid in enumerate(preds[:k]))
    ideal = sorted([score for _, score in truth], reverse=True)[:k]
    idcg = sum((2**score - 1) / np.log2(idx + 2) for idx, score in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0

# ========== BATCH EVALUATION ==========
results = []

for qid, query in tqdm(queries.items(), desc="Evaluating Queries"):
    q_embedding = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(q_embedding, clause_embeddings, top_k=TOP_K)[0]
    top_clause_ids = [clause_ids[hit['corpus_id']] for hit in hits]
    top_clause_texts = [corpus[cid] for cid in top_clause_ids]

    reranked = sorted(
        [(cid, gpt4o_score(query, corpus[cid])) for cid in top_clause_ids],
        key=lambda x: -x[1]
    )
    reranked_ids = [cid for cid, _ in reranked]

    truth = qrels_df[qrels_df['query_id'] == qid][['corpus_id', 'score']].values.tolist()

    ndcg5 = ndcg_at_k(reranked_ids, truth, 5)
    p3 = precision_at_k(reranked_ids, truth, 5, 3)
    p4 = precision_at_k(reranked_ids, truth, 5, 4)
    p5 = precision_at_k(reranked_ids, truth, 5, 5)

    results.append({
        "query_id": qid,
        "ndcg@5": ndcg5,
        "p@5_3star": p3,
        "p@5_4star": p4,
        "p@5_5star": p5
    })

results_df = pd.DataFrame(results)
results_df.to_csv("acord_clause_retrieval_results.csv", index=False)
print("Saved results to acord_clause_retrieval_results.csv")
