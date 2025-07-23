# search_module

Document management and hybrid search (semantic + keyword) module.

## Install
```bash
pip install .
```

## Usage
```python
from search_module import insert, search, delete

# Insert a file (PDF/TXT)
file_id = insert('path/to/file.pdf')

# Search
results = search('your query')

# Delete a file
delete(file_id)

# (Planned) Reindex all data if embedding model changes
# from search_module import reindex_all
# reindex_all()
```

## Features
- Insert PDF/TXT, auto-chunk, embed, and index
- Hybrid search: semantic (embedding) + keyword (TF-IDF)
- Delete all data for a file
- (Planned) Reindex all data if embedding model changes 