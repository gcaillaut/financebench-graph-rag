## Steps to reproduce

First, download the `financebench` dataset and extract text chunks from the pdf files.

```python
python download_financebench.py
python create_text_collection.py
```

Then, run the Text RAG using `python textrag.py`. It will generate a json file with the retrived chunks and the generated answer for each question in financebench.

From the output of the Text RAG, you can extract triples using the `text_chunks_to_graph.py`, which will scan the retrieved chunks and extract triples from them. It will also generact facts, which are text representation of triples.

Finally, the `textrag_fromfacts.py` script will query the RAG system using extracted facts instead of raw chunks.