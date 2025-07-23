from setuptools import setup, find_packages

setup(
    name='search_module',
    version='0.1.0',
    description='Document management and hybrid search (semantic + keyword) module',
    author='Your Name',
    packages=find_packages(),
    install_requires=[
        'sentence-transformers',
        'faiss-cpu',
        'pdfplumber',
        'scikit-learn',
        'numpy',
        'pyyaml',
    ],
    python_requires='>=3.7',
) 