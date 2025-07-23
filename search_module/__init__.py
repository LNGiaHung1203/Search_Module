"""
search_module: Document search and management package

Usage:
    from search_module import insert, search, delete, reindex_all
"""
from .document_manager import insert, delete
from .search_engine import search
# from .reindex import reindex_all  # To be implemented 