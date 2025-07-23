import os
import faiss
import numpy as np
import sqlite3
from .embedding import embed_text
from .utils import extract_and_split
import yaml

with open('search_module/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

VECTOR_STORE_PATH = config['vector_store_path']
META_STORE_PATH = config['meta_store_path']
RAW_FILES_DIR = config['raw_files_dir']
CHUNK_SIZE = config['chunk_size']

# Ensure storage dirs exist
os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
os.makedirs(RAW_FILES_DIR, exist_ok=True)

# FAISS setup (assume 384 dims for MiniLM)
if os.path.exists(VECTOR_STORE_PATH):
    index = faiss.read_index(VECTOR_STORE_PATH)
else:
    index = faiss.IndexIDMap(faiss.IndexFlatL2(384))

def get_next_file_id(ext):
    conn = sqlite3.connect(META_STORE_PATH)
    c = conn.cursor()
    prefix = 'txt' if ext == '.txt' else 'pdf'
    c.execute("SELECT file_id FROM metadata WHERE file_id LIKE ?", (f'{prefix}%',))
    ids = [row[0] for row in c.fetchall()]
    conn.close()
    nums = [int(fid[len(prefix):]) for fid in ids if fid.startswith(prefix) and fid[len(prefix):].isdigit()]
    next_num = max(nums) + 1 if nums else 1
    return f"{prefix}{next_num}"

def insert(file_path, file_id=None):
    ext = os.path.splitext(file_path)[1].lower()
    if not file_id:
        file_id = get_next_file_id(ext)
    conn = sqlite3.connect(META_STORE_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS metadata (
        file_id TEXT,
        chunk_id TEXT PRIMARY KEY,
        text TEXT,
        position INTEGER
    )''')
    # Save original file
    raw_dest = os.path.join(RAW_FILES_DIR, os.path.basename(file_path))
    if not os.path.exists(raw_dest):
        os.rename(file_path, raw_dest)
    chunks = extract_and_split(raw_dest, chunk_size=CHUNK_SIZE)
    vectors = embed_text(chunks)
    ids = np.array([i for i in range(len(chunks))], dtype=np.int64)
    chunk_ids = [f"{file_id}_{i}" for i in range(len(chunks))]
    index.add_with_ids(np.array(vectors).astype('float32'), ids)
    for i, chunk in enumerate(chunks):
        c.execute("INSERT INTO metadata (file_id, chunk_id, text, position) VALUES (?, ?, ?, ?)",
                  (file_id, chunk_ids[i], chunk, i))
    conn.commit()
    conn.close()
    faiss.write_index(index, VECTOR_STORE_PATH)
    return file_id

def delete(file_id):
    conn = sqlite3.connect(META_STORE_PATH)
    c = conn.cursor()
    c.execute("SELECT chunk_id, position FROM metadata WHERE file_id=?", (file_id,))
    rows = c.fetchall()
    chunk_indices = [row[1] for row in rows]
    if chunk_indices:
        index.remove_ids(np.array(chunk_indices, dtype=np.int64))
        faiss.write_index(index, VECTOR_STORE_PATH)
    c.execute("DELETE FROM metadata WHERE file_id=?", (file_id,))
    conn.commit()
    conn.close()
    # Optionally remove file
    # (not implemented: could scan RAW_FILES_DIR) 