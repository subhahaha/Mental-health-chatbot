"""
Build FAISS embedding indexes for the RAG knowledge base.

Takes each category's JSON file (anxiety.json, stress.json, etc.), embeds
every document's content using sentence-transformers, and builds a small
FAISS index per category - so retrieval can be scoped to exactly the
category the classifier predicted.

Run on Colab (needs Hugging Face access, same as train_distilbert.py):
!pip install sentence-transformers faiss-cpu -q
!python build_embeddings.py

Expects a knowledge_base/ folder (with the 5 JSON files we wrote together)
uploaded alongside this script.
"""

import json
import glob
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

KB_DIR = 'knowledge_base'
OUTPUT_DIR = 'knowledge_base_index'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Same embedding model used in DocMind - small, fast, well-suited for
# short text like these documents. Not the biggest/best available, but
# a good balance of speed and quality for a project this size.
MODEL_NAME = 'all-MiniLM-L6-v2'

print(f"Loading embedding model: {MODEL_NAME} ...")
model = SentenceTransformer(MODEL_NAME)
print("Model loaded.\n")

kb_files = glob.glob(f'{KB_DIR}/*.json')
print(f"Found {len(kb_files)} knowledge base files: {[os.path.basename(f) for f in kb_files]}\n")

for filepath in kb_files:
    with open(filepath) as f:
        data = json.load(f)

    category = data['category']
    documents = data['documents']
    safe_name = category.lower().replace(' ', '_')

    print(f"=== {category} ({len(documents)} documents) ===")

    # Embed the 'content' field of every document - this is the only
    # field that matters for retrieval, title/id are just metadata.
    contents = [doc['content'] for doc in documents]
    embeddings = model.encode(contents, convert_to_numpy=True)

    # Normalize vectors so we can use inner product as cosine similarity -
    # this measures how close two pieces of text are in MEANING, not
    # just shared words. Un-normalized inner product would be biased by
    # vector magnitude rather than pure direction/similarity.
    faiss.normalize_L2(embeddings)

    # IndexFlatIP = brute-force exact search using inner product.
    # With only 4 documents per category, this is overkill in terms of
    # what FAISS is built for (it shines with millions of vectors), but
    # using it here keeps the same pattern as if the knowledge base grows
    # much larger later, and matches what you'd use in DocMind-scale RAG.
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # Save the index itself
    faiss.write_index(index, f'{OUTPUT_DIR}/{safe_name}.index')

    # Save the metadata separately - FAISS only stores vectors and returns
    # positions (0, 1, 2...), so we need this side file to map a matched
    # position back to the actual id/title/content text.
    with open(f'{OUTPUT_DIR}/{safe_name}_metadata.json', 'w') as f:
        json.dump(documents, f, indent=2)

    print(f"Saved index + metadata for {category}\n")

print("=== All categories embedded and indexed ===\n")

# --- Quick sanity check: try a few real queries and see what gets retrieved ---
print("=== Sanity check: sample retrievals ===\n")

test_queries = [
    ("Anxiety", "my heart is racing and I can't stop worrying something bad will happen"),
    ("Anxiety", "this has been going on for a month, I can't sleep and I'm scared something is wrong with me"),
    ("Stress", "I have way too much to do and no idea where to even start"),
    ("Depression", "I don't want to talk to anyone lately, I just want to disappear"),
]

for category, query in test_queries:
    safe_name = category.lower().replace(' ', '_')
    index = faiss.read_index(f'{OUTPUT_DIR}/{safe_name}.index')
    with open(f'{OUTPUT_DIR}/{safe_name}_metadata.json') as f:
        metadata = json.load(f)

    query_vec = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)

    top_k = 1
    scores, indices = index.search(query_vec, top_k)
    best_match = metadata[indices[0][0]]

    print(f"Category: {category}")
    print(f"Query: \"{query}\"")
    print(f"Retrieved: \"{best_match['title']}\" (similarity: {scores[0][0]:.3f})")
    print()

print("If the retrieved documents above look sensible for each query, embedding worked correctly.")
print(f"\nDone. Index files saved in: {OUTPUT_DIR}/")
