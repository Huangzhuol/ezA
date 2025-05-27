#!/usr/bin/env python3
"""
Embed a JSONL corpus using OpenAI's embedding API.
Reads `maud_corpus.jsonl`, extracts each record's `_id` and `text`, batches requests
by token count, truncates overly long documents, and writes a JSON mapping of `_id` to embedding vectors.

Usage:
    python embed_maud_corpus.py maud_corpus.jsonl embeddings.json
"""
import os
import sys
import json
import jsonlines
import openai
import tiktoken

# -----------------------------
# Configuration
# -----------------------------
MODEL_NAME = "text-embedding-ada-002"
# Maximum total tokens per batch (under API limit of 300k)
MAX_BATCH_TOKENS = 200_000
# Maximum tokens per document (model context limit ~8191)
MAX_DOC_TOKENS = 8191
# Set your API key via environment variable
openai.api_key = os.getenv("OPENAI_API_KEY", "")

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


def batch_by_tokens(docs, model=MODEL_NAME, max_tokens=MAX_BATCH_TOKENS):
    """
    Yield batches of docs where cumulative token count does not exceed max_tokens.
    """
    encoder = tiktoken.encoding_for_model(model)
    batch = []
    token_count = 0
    for doc in docs:
        tok_len = len(encoder.encode(doc['text']))
        length = min(tok_len, MAX_DOC_TOKENS)
        if token_count + length > max_tokens and batch:
            yield batch
            batch = []
            token_count = 0
        batch.append(doc)
        token_count += length
    if batch:
        yield batch


def embed_texts(texts):
    """
    Send a batch of texts to OpenAI embedding API and return the list of embedding objects.
    """
    resp = openai.embeddings.create(
        model=MODEL_NAME,
        input=texts
    )
    # Return the list of Embedding objects
    return resp.data

# -----------------------------
# Main Routine
# -----------------------------

def main(corpus_path, output_path):
    # Verify API key
    if not openai.api_key:
        print("Error: OPENAI_API_KEY not set.")
        sys.exit(1)

    # Load documents
    docs = load_corpus(corpus_path)
    print(f"Loaded {len(docs)} documents from {corpus_path}.")

    # Prepare storage
    id_to_embedding = {}
    encoder = tiktoken.encoding_for_model(MODEL_NAME)

    # Create batches and process
    batches = list(batch_by_tokens(docs))
    total = len(batches)
    for idx, batch in enumerate(batches, start=1):
        # Truncate each doc to MAX_DOC_TOKENS tokens
        texts = []
        for doc in batch:
            tokens = encoder.encode(doc['text'])
            if len(tokens) > MAX_DOC_TOKENS:
                tokens = tokens[:MAX_DOC_TOKENS]
            texts.append(encoder.decode(tokens))

        embeddings = embed_texts(texts)
        # Map embeddings back to document IDs
        for j, entry in enumerate(embeddings):
            doc_id = batch[j]['_id']
            # entry is an Embedding object with .embedding attribute
            id_to_embedding[doc_id] = entry.embedding

        print(f"Processed batch {idx}/{total}: {len(batch)} docs")

    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as fout:
        json.dump(id_to_embedding, fout, ensure_ascii=False)

    print(f"Saved embeddings for {len(id_to_embedding)} documents to {output_path}")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python embed_maud_corpus.py <input.jsonl> <output.json>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
