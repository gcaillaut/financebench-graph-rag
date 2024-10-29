from llmworkflow import (
    Workflow,
    TextVectorStore,
    ChatModel,
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
            "content": f"Please read the text passages below, I will ask you questions afterward.\n\n{passages_text}",
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

    llm = ChatModel("chat_model", config["text_model"], config["max_new_tokens"])

    def _make_prompt(query):
        passages = [
            x["content"]
            for x in text_vector_store.search(query["question"], where={"path": {"$eq": str(query["path"])}})
        ]
        
        return {
            "question": query["question"],
            # "question": f"{query['question']}\n\nTo answer this question, first describe you reasoning process betwen <reasoning></reasoning> XML tags. Then, provide your definitive answer between <answer></answer> tags. The definitive answer should be short and clear.",
            "history": get_base_history(passages),
            "chunks": passages,
        }

    def _ask_llm(data):
        raw_answer = llm(data["question"], data["history"])
        return {
            "prediction": raw_answer,
            # "prediction": extract_xml_tag(raw_answer, "answer"),
            # "llm_explanation": extract_xml_tag(raw_answer, "reasoning"),
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
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    output_path = "output/financebench_text_results_llama3.2-3B.json"
    
    # model_name = "meta-llama/Llama-3.1-8B-Instruct"
    # output_path = "financebench_facts_results_llama3.1-8B.json"
    
    
    config = {
        "chroma_db_path": "./cache/chromadb",
        "text_index_name": "text_financebench",
        "embedding_model": "all-MiniLM-L6-v2",
        "top_k_results": 8,
        "text_model": model_name,
        "max_new_tokens": 2048,
    }

    rag_workflow = create_rag_workflow(config)

    ds = datasets.load_dataset("PatronusAI/financebench", split="train")

    docid = ds["financebench_id"]
    questions = ds["question"]
    answers = ds["answer"]
    justifications = ds["justification"]

    results = []
    for id, q, a, j in tqdm(
        zip(docid, questions, answers, justifications), total=len(ds)
    ):
        pdf_path = Path("data", "financebench", f"{id}.pdf")
        if pdf_path.is_file():

            out = rag_workflow.run(
                {"question": q, "path": Path("data", "financebench", f"{id}.pdf")}
            )
            results.append(
                {
                    "question": q,
                    "expected": a,
                    "predicted": out["prediction"],
                    # "llm_justification": out["llm_explanation"],
                    "justification": j,
                    "chunks": out["chunks"],
                }
            )
        break

    with open(output_path, "wt", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
