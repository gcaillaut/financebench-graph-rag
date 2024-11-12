from llmworkflow import (
    Workflow,
    TextVectorStore,
    ChatModel,
    ApiModel,
    Chain,
    Function,
)
import datasets
from pathlib import Path
from tqdm import tqdm
import re
import json


def get_base_history(passages):
    passages_text = "\n\n\n".join(
        f"### Passage {i}\n\n{p}" for i, p in enumerate(passages, start=1)
    )
    return [
        {
            "role": "system",
            "content": "You are a friendly assistant. Provide useful and clear answers. Be concise. Do not try to do calculus, rather get the information from the context.",
        },
        {
            "role": "user",
            "content": f"Below are triples from a knowledge graph, please read them. I will ask you questions afterward.\n\n{passages_text}",
        },
        {
            "role": "assistant",
            "content": "I have read the text passages, I am ready to answer your questions.",
        },
    ]
    
def extract_xml_tag(text, tag):
    pattern = rf"<{tag}>(.+?)</{tag}>"
    m = re.match(pattern, text, re.DOTALL | re.IGNORECASE)
    
    return m.group(0)


def create_rag_workflow(config):
    rag_workflow = Workflow("Text RAG")

    text_vector_store = TextVectorStore(
        "text_vector_store",
        config["text_index_name"],
        config["chroma_db_path"],
        config["embedding_model"],
        config["top_k_results"],
    )

    if config["api"]:
        llm = ApiModel("chat_model", config["text_model"], "http://localhost:8000/v1", "token", config["max_new_tokens"])
    else:
        llm = ChatModel("chat_model", config["text_model"], config["max_new_tokens"])

    def _make_prompt(query):
        passages = [
            x["metadata"]["triples"].strip()
            for x in text_vector_store.search(query["question"], where={"document": {"$eq": query["document"]}})
        ]
        
        return {
            "question": query["question"],
            "history": get_base_history(passages),
            "chunks": passages,
        }

    def _ask_llm(data):
        try:
            raw_answer = llm(data["question"], data["history"])
        except:
            raw_answer = ""
        # raw_answer = "\n\n-------------\n\n".join(x["content"] for x in data["history"]) + "\n\n\n\n\n" + data["question"]
        return {
            "prediction": raw_answer,
            "chunks": data["chunks"],
        }

    rag_chain = Chain(
        "chain all",
        [
            Function("make_prompt", _make_prompt),
            Function("ask_llm", _ask_llm),
        ],
    )
    rag_workflow.add(rag_chain)
    return rag_workflow

if __name__ == "__main__":
    # model_name = "meta-llama/Llama-3.2-3B-Instruct"
    # output_path = "output/financebench_neo4j-no-text_results_llama3.2-3B.json"
    
    # model_name = "meta-llama/Llama-3.1-8B-Instruct"
    # output_path = "output/financebench_neo4j-no-text_results_llama3.1-8B.json"
    
    model_name = "Qwen/Qwen2.5-32B-Instruct"
    output_path = "output/financebench_neo4j-no-text_results_qwen2.5-32B.json"
    
    # model_name = "Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int8"
    # output_path = "output/financebench_neo4j-no-text_results_qwen2-vl-72B.json"
    
    config = {
        "chroma_db_path": "./cache/chromadb",
        "text_index_name": "neo4j_financebench",
        "embedding_model": "all-MiniLM-L6-v2",
        "top_k_results": 8,
        "text_model": model_name,
        "max_new_tokens": 2048,
        "api": model_name == "Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int8",
    }

    rag_workflow = create_rag_workflow(config)

    ds = datasets.load_dataset("PatronusAI/financebench", split="train")

    docnames = ds["doc_name"]
    questions = ds["question"]
    answers = ds["answer"]
    justifications = ds["justification"]

    results = []
    for dn, q, a, j in tqdm(
        zip(docnames, questions, answers, justifications), total=len(ds)
    ):
        pdf_path = Path("data", "financebench", f"{dn}.pdf")
        if pdf_path.is_file():
            out = rag_workflow.run(
                {"question": q, "document": dn}
            )
            results.append(
                {
                    "question": q,
                    "expected": a,
                    "predicted": out["prediction"],
                    "justification": j,
                    "chunks": out["chunks"],
                }
            )

    with open(output_path, "wt", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
