import os
import json
import torch
import torch.nn.functional as F
import jsonlines
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY", "")
if not api_key:
    raise ValueError("Missing OPENAI_API_KEY environment variable")
client = OpenAI(api_key=api_key)


def load_embeddings(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict):
        ids = list(data.keys())
        embs = torch.tensor(list(data.values()), dtype=torch.float)
    elif isinstance(data, list):
        ids = [item['_id'] for item in data]
        embs = torch.tensor([item['embedding'] for item in data], dtype=torch.float)
    else:
        raise ValueError(f"Unsupported embedding format: {type(data)}")

    embs = F.normalize(embs, p=2, dim=1)
    return ids, embs


def get_query_emb_openai(query: str, model: str = "text-embedding-ada-002"):
    resp = client.embeddings.create(model=model, input=[query])
    vec = resp.data[0].embedding
    t = torch.tensor(vec, dtype=torch.float)
    return F.normalize(t, p=2, dim=0)


def retrieve_similar(query: str,
                     corpus: dict,
                     embedding_path: str,
                     top_k: int = 5):
    ids, embs = load_embeddings(embedding_path)
    q_emb = get_query_emb_openai(query)
    cos_scores = torch.mm(q_emb.unsqueeze(0), embs.t())[0]
    top = torch.topk(cos_scores, k=top_k)
    results = []
    for score, idx in zip(top.values, top.indices):
        cid = ids[idx]
        results.append({
            "id": cid,
            "text": corpus.get(cid, ""),
            "score": float(score)
        })
    return results


def load_corpus(path: str):
    d = {}
    with jsonlines.open(path, "r") as reader:
        for item in reader:
            d[item.get("_id")] = item.get("text", "")
    return d


def ragFunc(corpus_path: str = "maud_corpus.jsonl",
         embedding_path: str = "embeddings.json",
         query: str = None,
         top_k: int = 2):
    if query is None:
        raise ValueError("Please provide a 'query' to main()")
    corpus = load_corpus(corpus_path)
    return retrieve_similar(query, corpus, embedding_path, top_k)


if __name__ == "__main__":
    query_input = "put ur query"
    results = ragFunc(query=query_input)
    for m in results:
        print(f"→ {m['id']} (score={m['score']:.4f}):\n{m['text']}\n")
