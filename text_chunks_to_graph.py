from llmworkflow import (
    Workflow,
    ChatModel,
    Chain,
    Function,
)
import json
import random
from tqdm import tqdm

from transformers.cache_utils import (
    DynamicCache,
    SinkCache,
    StaticCache,
    SlidingWindowCache,
    QuantoQuantizedCache,
    QuantizedCacheConfig,
)


def make_assistant_turn(text):
    return {"role": "assistant", "content": text}

def make_user_turn(text):
    return {"role": "user", "content": text}

def get_json_list(text):
    i = text.find("[")
    j = text.rfind("]")
    
    if any((i < 0, j < 0)):
        raise ValueError(f"Not a JSON list: \n{text}")
    
    return json.loads(text[i:(j+1)])

def augment_ner_prompt(base_prompt, facts, max_facts=8):
    if facts:
        if len(facts) > max_facts:
            facts = random.sample(facts, max_facts)
        # return f"{base_prompt}\n\nThe text above is the continuation of a document I process iteratively. I have already extracted some triples, which are: \n\n```json\n{json.dumps(triples, ensure_ascii=False, indent=2)}\n```\n\nUse them to avoid generating duplicates and to remove ambiguity in the text."
    
        return f"{base_prompt}\n\nThe text above is the continuation of a document I process iteratively. I have already extracted some facts, which are: \n\n{'\n'.join('- ' + f for f in facts)}\n\n\nUse them to avoid generating duplicates and to remove ambiguity in the text."
    else:
        return base_prompt
    
def create_knowledge_graph_workflow(config):
    workflow = Workflow("KG extractor")
    llm = ChatModel("LLM", config["text_model"], config["max_new_tokens"])
    
    ner_prompt = "The end goal is to build a knowledge graph from the text. We will do it step by step. First, extract all named entities (persons, organizations, events, ...), dates (times and epochs too) and locations. Put them in a list."
    
    triples_prompt = "Perfect, now generate a list of triples (subject, predicate, object). Subjects and objects must come either from the list of entities you extracted beforehand, or the list of triples I might have provided. Predicates are very short text (up to 3 words) describing the relation between subjects and objects. Try to extract only *interesting* triples, do not report too obvious triples."
    
    json_prompt = "Great, now format the triples as a JSON list. Add a \"text\" attribute containing a sentence in natural language fully describing the fact held by the triple. Just write the JSON content."
    
    def print_last_messages(h, header):
        print(header)
        print(h[-1]["content"])
        return h
    
    generation_kwargs = dict(top_p=None, top_k=None, temperature=None, do_sample=False, num_beams=1)
    
    def ner_step(history, known_facts):
        past_key_values = None
        # past_key_values = DynamicCache()
        # past_key_values = StaticCache(llm.model.config, batch_size=1, device=llm.model.device)
        augmented_ner_prompt = augment_ner_prompt(ner_prompt, known_facts)
        response = llm(augmented_ner_prompt, history, past_key_values=past_key_values, **generation_kwargs)
        return history + [make_user_turn(augmented_ner_prompt), make_assistant_turn(response)], past_key_values
    
    def triples_step(history, past_key_values):
        response = llm(triples_prompt, history, past_key_values=past_key_values, **generation_kwargs)
        return history + [make_user_turn(triples_prompt), make_assistant_turn(response)], past_key_values
    
    def json_step(history, past_key_values):
        response = llm(json_prompt, history, past_key_values=past_key_values, **generation_kwargs)
        return response

    chain = Chain("extract", [
        # data is a tuple (history, list of facts)
        Function("NER", lambda data: ner_step(*data)),
        # Function("Print NER", partial(print_last_messages, header = "STEP 1")),
        Function("REL", lambda data: triples_step(*data)),
        # Function("Print REL", partial(print_last_messages, header = "STEP 2")),
        Function("JSON", lambda data: json_step(*data)),
        Function("TOJSON", get_json_list),
    ])
    
    workflow.add(chain)
    return workflow



XP_NAMES = [
    # "financebench_text_results_llama3.1-8B",
    # "financebench_text_results_llama3.2-3B",
    "financebench_text_results_qwen2.5-32B",
]

TEXT_MODELS = [
    # "meta-llama/Llama-3.1-8B-Instruct",
    # "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
]

for xp_name, txt_model in zip(XP_NAMES, TEXT_MODELS):
    with open(f"output/{xp_name}.json", "rt", encoding="utf-8") as f:
        text_data = json.load(f)


    config = {
        "text_model": txt_model,
        "max_new_tokens": 2048,
    }

    kg_workflow = create_knowledge_graph_workflow(config)


    for txt_results in tqdm(text_data):
        txt_results["graph"] = {"triples": [], "facts": []}
        chunks = txt_results["chunks"]
        w = 2
        known_facts = []
        for i in range(0, len(chunks), w):
            text = "\n".join(chunks[i:(i+w)])
            
            base_history = [
                {"role": "system", "content": "You are a friendly assistant. Provide useful and clear answers."},
                {"role": "user", "content": f"Please read the text below, I will ask you questions afterward.\n\n{text}"},
                {"role": "assistant", "content": "I have read the text, I am ready to answer your questions."},
            ]
            
            try:
                result = kg_workflow.run((base_history, known_facts))
            
                known_facts.extend(x.get("text", "") for x in result)
                known_facts = list(set(known_facts))
            
                txt_results["graph"]["triples"].extend(result)
            except ValueError:
                pass
        txt_results["graph"]["facts"] = known_facts


    with open(f"output/{xp_name}_with-facts.json", "wt", encoding="utf-8") as f:
        json.dump(text_data, f, indent=4, ensure_ascii=False)
