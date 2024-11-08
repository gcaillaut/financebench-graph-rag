import json
from transformers import AutoTokenizer

def get_textrag_base_history(passages):
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
    
def make_textrag_prompt(question, passages):
    history = get_textrag_base_history(passages)
    return history + [{"role": "user", "content": question}]

def get_factsrag_base_history(passages):
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
    
def make_factsrag_prompt(question, facts):
    history= get_factsrag_base_history(facts)
    return history + [{"role": "user", "content": question}]

def read_json(path):
    with open(path, "rt", encoding="utf-8") as f:
        return json.load(f)

def count_tokens(tokenizer, messages):
    tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    return len(tokens)


if __name__ == "__main__":
    
    xp_names = ["llama3.1-8B", "llama3.2-3B", "qwen2.5-32B"]
    model_names = [
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
    ]
    
    output = []

    for xp, model in zip(xp_names, model_names):
        tokenizer = AutoTokenizer.from_pretrained(model)

        textrag_output_path = f"output/financebench_text_results_{xp}.json"
        factsrag_output_path = f"output/financebench_facts_results_{xp}.json"
        neo4jrag_output_path = f"output/financebench_neo4j_results_{xp}.json"
            
        textrag_data = read_json(textrag_output_path)
        factsrag_data = read_json(factsrag_output_path)
        neo4jrag_data = read_json(neo4jrag_output_path)
                
        textrag_tokens = 0
        factsrag_tokens = 0
        neo4jrag_tokens = 0

        for x in textrag_data:
            messages = make_textrag_prompt(x["question"], x["chunks"])
            textrag_tokens += count_tokens(tokenizer, messages)
            
        for x in factsrag_data:
            messages = make_factsrag_prompt(x["question"], x["chunks"])
            factsrag_tokens += count_tokens(tokenizer, messages)
            
        for x in neo4jrag_data:
            messages = make_factsrag_prompt(x["question"], x["chunks"])
            neo4jrag_tokens += count_tokens(tokenizer, messages)
            
        output.append({
            "model": model,
            "facts": factsrag_tokens,
            "text": textrag_tokens,
            "neo4j": neo4jrag_tokens,
        })
        
    with open("output/input_tokens_count.json", "wt", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)