from typing import Union, List, Tuple, Dict
from .core import Node
import itertools
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)


class TextModel(Node):
    def __init__(self, name: str, model_name: str, max_new_tokens: int = 1024, batch_size: int = 8):
        super().__init__(name)
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation = "sdpa", torch_dtype=torch.bfloat16, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, prompt: Union[str, List[str]], **kwargs) -> Union[str, List[str]]:
        if isinstance(prompt, str):
            prompts = [prompt]
            return_list = False
        elif isinstance(prompt, list) and isinstance(prompt[0], str):
            prompts = prompt
            return_list = True
        else:
            raise ValueError("Expected input to be a string or a list of strings.")

        generated_text = list(itertools.chain.from_iterable(
            self._generate(prompts[i:(i+self.batch_size)], **kwargs)
            for i in range(0, len(prompts), self.batch_size)
        ))
        return generated_text if return_list else generated_text[0]
    
    def _generate(self, batch, **kwargs):
        inputs = self._tokenize(batch)

        input_length = inputs["input_ids"].size(-1)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, **kwargs)
        return self.tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=True)
    
    def _tokenize(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        for k, v in inputs.items():
            inputs[k] = v.to(self.model.device)
        return inputs

    def get_tokenizer(self):
        return self.tokenizer

    def get_logits(self, text: str):
        inputs = self._tokenizer(text)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = self.model(**inputs)
        logits = outputs.logits
        return logits


class ChatModel(TextModel):
    def __call__(self,
                 prompts: Union[str, List[str]],
                 histories: Union[List[Dict[str, str]], List[List[Dict[str, str]]]] = None,
                 **kwargs) -> Union[str, List[str]]:
        return_list = True
        if isinstance(prompts, str):
            return_list = False
            prompts = [prompts]
            if histories is None:
                histories = [[]]
            else:
                histories = [histories]
        elif histories is None:
            histories = [[] for _ in prompts]
            
        full_prompts = [h + [{"role": "user", "content": p}] for h, p in zip(histories, prompts)]
        texts = self.tokenizer.apply_chat_template(full_prompts, add_generation_prompt=True, tokenize=False)
        return super().__call__(texts if return_list else texts[0], **kwargs)
    
    def get_logits(self, prompt: str, history: List[Dict[str, str]]):
        full_prompt = history + [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(full_prompt, tokenize=False)[0]
        return super().get_logits(text)