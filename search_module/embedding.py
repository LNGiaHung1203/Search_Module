from sentence_transformers import SentenceTransformer
import yaml

with open('search_module/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

model = SentenceTransformer(config['embedding_model'])

def embed_text(texts):
    """
    Embed a list of texts using the configured model.
    Args:
        texts (List[str]): List of text strings.
    Returns:
        np.ndarray: Embedding vectors.
    """
    return model.encode(texts, show_progress_bar=False) 