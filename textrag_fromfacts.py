from llmworkflow import (
    Workflow,
    ChatModel,
    Chain,
    Function,
)
import datasets
import json
from tqdm import tqdm
import re


def get_base_history(passages):
    passages_text = "\n".join(
        f"- {p}" for p in passages
    )
    return [
        {
            "role": "system",
            "content": "You are a friendly assistant. Provide useful and clear answers. Be concise. Do not try to do calculus, rather get the information from the context.",
        },
        {
            "role": "user",
            "content": f"Please read the facts below, I will ask you questions afterward.\n\n{passages_text}",
        },
        {
            "role": "assistant",
            "content": "I have read and understood the facts, I am ready to answer your questions.",
        },
    ]
    
def extract_xml_tag(text, tag):
    pattern = rf"<{tag}>(.+?)</{tag}>"
    m = re.match(pattern, text, re.DOTALL | re.IGNORECASE)
    
    return m.group(0)


def create_rag_workflow(config):
    rag_workflow = Workflow("Text RAG")

    llm = ChatModel("chat_model", config["text_model"], config["max_new_tokens"])

    def _make_prompt(query):
        return {
            "question": query["question"],
            # "question": f"{query['question']}\n\nTo answer this question, first describe you reasoning process betwen <reasoning></reasoning> XML tags. Then, provide your definitive answer between <answer></answer> tags. The definitive answer should be short and clear.",
            "history": get_base_history(query["facts"]),
        }

    def _ask_llm(data):
        raw_answer = llm(data["question"], data["history"])
        return {
            "prediction": raw_answer,
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
    # json_path = "output/financebench_text_results_llama3.2-3B_with-facts.json"
    # output_path = "financebench_facts_results_llama3.2-3B.json"
    
    # model_name = "meta-llama/Llama-3.1-8B-Instruct"
    # json_path = "financebench_text_results_llama3.1-8B_with-facts.json"
    # output_path = "financebench_facts_results_llama3.1-8B.json"
    
    model_name = "Qwen/Qwen2.5-32B-Instruct"
    json_path = "financebench_text_results_qwen2.5-32B_with-facts.json"
    output_path = "financebench_facts_results_qwen2.5-32B.json"
    
    config = {
        "text_model": model_name,
        "max_new_tokens": 2048,
    }

    rag_workflow = create_rag_workflow(config)

    ds = datasets.load_dataset("PatronusAI/financebench", split="train")
    
    with open(json_path, "rt", encoding="utf-8") as f:
        ds = json.load(f)

    results = []
    for x in tqdm(ds):
        q = x["question"]
        out = rag_workflow.run(
            {"question": q, "facts": x["graph"]["facts"]}
        )
        results.append(
            {
                "question": q,
                "expected": x["expected"],
                "predicted": out["prediction"],
                # "llm_justification": out["llm_explanation"],
                "justification": x["justification"],
                "chunks": x["graph"]["facts"],
                "raw_text_chunks": x["chunks"],
                "graph": x["graph"],
            }
        )

    with open(output_path, "wt", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
