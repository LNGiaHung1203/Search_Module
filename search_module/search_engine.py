import faiss
import numpy as np
import sqlite3
from .embedding import embed_text
from .utils import generate_chunk_id
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

with open('search_module/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

VECTOR_STORE_PATH = config['vector_store_path']
META_STORE_PATH = config['meta_store_path']
ALPHA = config['search']['alpha']
BETA = config['search']['beta']
TOP_K = config['search']['top_k']

# FAISS with optimized index
if os.path.exists(VECTOR_STORE_PATH):
    index = faiss.read_index(VECTOR_STORE_PATH)
else:
    # Use IndexFlatIP (inner product) for better performance with normalized vectors
    index = faiss.IndexIDMap(faiss.IndexFlatIP(384))

# Cache for TF-IDF vectorizer to avoid recomputation
_tfidf_cache = {'vectorizer': None, 'matrix': None, 'chunk_texts_hash': None}

def semantic_search(query, chunk_ids, chunk_texts, chunk_meta, top_k=TOP_K):
    q_vec = embed_text([query])[0].astype('float32')
    # Normalize query vector for cosine similarity with IndexFlatIP
    faiss.normalize_L2(q_vec.reshape(1, -1))
    
    D, I = index.search(q_vec.reshape(1, -1), top_k)
    results = []
    valid_indices = I[0] != -1
    valid_I = I[0][valid_indices]
    valid_D = D[0][valid_indices]
    
    # Vectorized bounds checking
    valid_mask = (valid_I >= 0) & (valid_I < len(chunk_ids))
    valid_I = valid_I[valid_mask]
    valid_D = valid_D[valid_mask]
    
    for idx, dist in zip(valid_I, valid_D):
        results.append({
            'chunk_id': chunk_ids[idx],
            'score': float(dist),  # Convert to Python float for JSON serialization
            'file_id': chunk_meta[idx][0],
            'position': chunk_meta[idx][1],
            'text': chunk_texts[idx],
        })
    return results

def keyword_search(query, chunk_ids, chunk_texts, chunk_meta, top_k=TOP_K):
    if not chunk_texts:
        return []
    
    global _tfidf_cache
    chunk_texts_hash = hash(tuple(chunk_texts))
    
    # Use cached TF-IDF matrix if chunk_texts haven't changed
    if (_tfidf_cache['chunk_texts_hash'] == chunk_texts_hash and 
        _tfidf_cache['vectorizer'] is not None):
        vectorizer = _tfidf_cache['vectorizer']
        tfidf_matrix = _tfidf_cache['matrix']
    else:
        # Optimized TF-IDF parameters
        vectorizer = TfidfVectorizer(
            max_features=10000,  # Limit vocabulary size
            stop_words='english',  # Remove common stop words
            ngram_range=(1, 2),  # Include bigrams for better context
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.8  # Ignore terms that appear in more than 80% of documents
        )
        tfidf_matrix = vectorizer.fit_transform(chunk_texts)
        
        # Cache the results
        _tfidf_cache = {
            'vectorizer': vectorizer,
            'matrix': tfidf_matrix,
            'chunk_texts_hash': chunk_texts_hash
        }
    
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, tfidf_matrix)[0]
    
    # Use argpartition for better performance when top_k << len(sims)
    if top_k < len(sims) // 10:
        top_idx = np.argpartition(sims, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
    else:
        top_idx = np.argsort(sims)[::-1][:top_k]
    
    results = []
    for idx in top_idx:
        if sims[idx] > 0:  # Only include results with positive similarity
            results.append({
                'chunk_id': chunk_ids[idx],
                'score': float(sims[idx]),  # Convert to Python float for JSON serialization
                'file_id': chunk_meta[idx][0],
                'position': chunk_meta[idx][1],
                'text': chunk_texts[idx],
            })
    return results

def search(query, top_k=TOP_K):
    conn = sqlite3.connect(META_STORE_PATH)
    c = conn.cursor()
    c.execute("SELECT chunk_id, text, file_id, position FROM metadata")
    rows = c.fetchall()
    conn.close()
    
    chunk_texts = [row[1] for row in rows]
    chunk_ids = [row[0] for row in rows]
    chunk_meta = [(row[2], row[3]) for row in rows]  # (file_id, position)
    
    # Adaptive top_k multiplier based on collection size
    multiplier = min(3, max(2, len(chunk_texts) // 1000 + 2))
    search_k = min(top_k * multiplier, len(chunk_texts))
    
    sem_results = semantic_search(query, chunk_ids, chunk_texts, chunk_meta, top_k=search_k)
    key_results = keyword_search(query, chunk_ids, chunk_texts, chunk_meta, top_k=search_k)
    
    # Optimized merging with dictionary comprehension
    scores = {r['chunk_id']: {'semantic': float(r['score']), 'keyword': 0.0, 'meta': r} 
              for r in sem_results}
    
    for r in key_results:
        if r['chunk_id'] in scores:
            scores[r['chunk_id']]['keyword'] = float(r['score'])
        else:
            scores[r['chunk_id']] = {'semantic': 0.0, 'keyword': float(r['score']), 'meta': r}
    
    # Normalize scores for better combination
    if scores:
        sem_scores = [v['semantic'] for v in scores.values()]
        key_scores = [v['keyword'] for v in scores.values()]
        
        # Robust normalization (handles edge cases)
        sem_max, sem_min = max(sem_scores), min(sem_scores)
        key_max, key_min = max(key_scores), min(key_scores)
        
        sem_range = sem_max - sem_min if sem_max != sem_min else 1
        key_range = key_max - key_min if key_max != key_min else 1
        
        # Vectorized score combination
        combined = []
        for cid, v in scores.items():
            # Normalize scores to [0, 1] range
            norm_semantic = (v['semantic'] - sem_min) / sem_range
            norm_keyword = (v['keyword'] - key_min) / key_range
            
            combined_score = float(ALPHA * norm_keyword + BETA * norm_semantic)
            meta = v['meta']
            meta['combined_score'] = combined_score
            combined.append(meta)
        
        # Use partial sort for better performance
        if top_k < len(combined) // 4:
            combined = sorted(combined, key=lambda x: x['combined_score'], reverse=True)[:top_k]
        else:
            combined.sort(key=lambda x: x['combined_score'], reverse=True)
            combined = combined[:top_k]
    else:
        combined = []
    
    return combined