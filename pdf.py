from unstructured.partition.pdf import partition_pdf


def extract_chunks(pdf_path, chunk_size=500):
    elements = partition_pdf(pdf_path, url=None, infer_table_structure=False)

    chunks = []
    current = []
    current_len = 0

    for e in elements:
        e_str = str(e)
        l = len(e_str)

        if current_len + l < chunk_size:
            current.append(e_str)
            current_len += l
        else:
            chunks.append("\n".join(current))
            current = [e_str]
            current_len = l

    if len(current) > 0:
        chunks.append("\n".join(chunks))

    return chunks
