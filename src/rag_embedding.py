#!/usr/bin/env python3
"""
Embed a JSONL corpus using OpenAI's embedding API.
Reads `maud_corpus.jsonl`, extracts each record's `_id` and `text`, batches requests,
and writes a JSON mapping of `_id` to embedding vectors.

Usage:
    python embed_maud_corpus.py maud_corpus.jsonl embeddings.json
"""
import os
import sys
import json
import jsonlines
import openai

# -----------------------------
# Configuration
# -----------------------------
MODEL_NAME = "text-embedding-ada-002"
BATCH_SIZE = 500

# -----------------------------
# Functions
# -----------------------------
def load_corpus(jsonl_path):
    """
    Load a JSONL file and return list of dicts with keys '_id' and 'text'.
    """
    docs = []
    with jsonlines.open(jsonl_path) as reader:
        for obj in reader:
            if '_id' in obj and 'text' in obj:
                docs.append({'_id': str(obj['_id']), 'text': obj['text']})
    return docs


def embed_texts(texts):
    """
    Send a batch of texts to OpenAI embedding API and return the response list.
    """
    resp = openai.embeddings.create(
        model=MODEL_NAME,
        input=texts
    )
    return resp['data']

# -----------------------------
# Main Routine
# -----------------------------

def main(corpus_path, output_path):
    # Ensure API key is set
    api_key = "T3BlbkFJ9uNLRH5xWRjV52R8pdQ4i9h8k0R9IFb3AY0ylWBazrlGoErctKmxufGqiBK0eFLMF1nCo6WS8A"
    if not os.getenv(api_key):
        print("Error: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    openai.api_key = os.getenv(api_key)

    # Load documents
    docs = load_corpus(corpus_path)
    print(f"Loaded {len(docs)} documents from {corpus_path}.")

    # Prepare embedding storage
    id_to_embedding = {}

    # Process in batches
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i:i + BATCH_SIZE]
        texts = [d['text'] for d in batch]
        embeddings = embed_texts(texts)

        # Map embeddings back to document IDs
        for j, entry in enumerate(embeddings):
            doc_id = batch[j]['_id']
            id_to_embedding[doc_id] = entry['embedding']

        print(f"Processed batch {i // BATCH_SIZE + 1} / {((len(docs)-1) // BATCH_SIZE) + 1}")

    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as fout:
        json.dump(id_to_embedding, fout, ensure_ascii=False)

    print(f"Saved embeddings for {len(id_to_embedding)} documents to {output_path}")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python embed_maud_corpus.py <input.jsonl> <output.json>")
        sys.exit(1)
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    main(in_path, out_path)
