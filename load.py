from dataclasses import dataclass

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizer, TextStreamer

assert transformers.__version__ >= "4.34.1"

@dataclass
class Model:
    repo_id: str
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    streamer: TextStreamer
    
    @property
    def device(self):
        return self.model.device



_loaded_model: Model|None = None

def load_model(repo_id: str, model_args: dict, tokenizer_args: dict) -> Model:
    model_args["cache_dir"] = "./models"
    tokenizer_args["cache_dir"] = "./models"

    model = AutoModelForCausalLM.from_pretrained(repo_id, **model_args)
    tokenizer = AutoTokenizer.from_pretrained(repo_id, **tokenizer_args)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    print(f"model loaded: {repo_id}")
    return Model(repo_id, model, tokenizer, streamer)



def reload_model(repo_id: str, model_args: dict, tokenizer_args: dict):
    global _loaded_model
    if _loaded_model is None or _loaded_model.repo_id != repo_id:
        print(f"reload model: {repo_id}")
        _loaded_model = load_model(repo_id, model_args, tokenizer_args)



def reload_default_model():
    repo_id = "TheBloke/calm2-7B-chat-GPTQ"
    model_args = {
        "use_cache": True,
        "device_map": "auto",
        "revision": "gptq-4bit-32g-actorder_True",
    }
    tokenizer_args = {
        "revision": "gptq-4bit-32g-actorder_True",
    }
    reload_model(repo_id, model_args, tokenizer_args)



def get_model() -> Model:
    if _loaded_model is None:
        reload_default_model()
    return _loaded_model
