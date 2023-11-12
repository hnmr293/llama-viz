from dataclasses import dataclass

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizer, TextStreamer

assert transformers.__version__ >= "4.34.1"

@dataclass
class Model:
    repo_id: str
    revision: str|None
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    streamer: TextStreamer
    
    @property
    def device(self):
        return self.model.device



_loaded_model: Model|None = None

def load_model(repo_id: str, revision: str|None, model_args: dict, tokenizer_args: dict) -> Model:
    import time
    
    # load model
    
    print("start model loading...")
    t0 = time.perf_counter_ns()

    model = AutoModelForCausalLM.from_pretrained(repo_id, **model_args)
    tokenizer = AutoTokenizer.from_pretrained(repo_id, **tokenizer_args)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    t1 = time.perf_counter_ns()
    print(f"model loaded: {repo_id} [{(t1-t0)/1000000:.1f}ms]")
    
    ## compile model
    #
    #print("compiling...")
    #t2 = time.perf_counter_ns()
    #
    #model.model = torch.compile(model.model, dynamic=True)
    #
    #t3 = time.perf_counter_ns()
    #print(f"finish compiling [{(t3-t2)/1000000:.1f}ms]")
    
    return Model(repo_id, revision, model, tokenizer, streamer)



def reload_model_hf(repo_id: str, rev: str|None, model_args: dict, tokenizer_args: dict):
    global _loaded_model
    
    if rev == "":
        rev = None
    
    if "revision" in model_args or "revison" in tokenizer_args:
        if rev is not None:
            print(f'[WARN] `revision` in dict will be overrided with "{rev}"')
        else:
            rev = model_args.get("revision", None) or tokenizer_args["revision"]
    
    model_args["revision"] = rev
    tokenizer_args["revision"] = rev
    
    if _loaded_model is None or _loaded_model.repo_id != repo_id or _loaded_model.revision != rev:
        s = f"reload model: {repo_id}"
        if rev is not None:
            s += f" [{rev}]"
        print(s)
        print("with model args =", model_args)
        print(" tokenizer args =", tokenizer_args)

        if "cache_dir" not in model_args or "cache_dir" not in tokenizer_args:
            print("[WARN] cache_dir is not specified")

        _loaded_model = load_model(repo_id, rev, model_args, tokenizer_args)
    return _loaded_model



def reload_default_model():
    repo_id = "TheBloke/calm2-7B-chat-GPTQ"
    model_args = {
        "use_cache": True,
        "device_map": "auto",
        "revision": "gptq-4bit-32g-actorder_True",
        "cache_dir": "./models",
    }
    tokenizer_args = {
        "revision": "gptq-4bit-32g-actorder_True",
        "cache_dir": "./models",
    }
    return reload_model_hf(repo_id, model_args, tokenizer_args)



def get_loaded_model_or_default() -> Model:
    if _loaded_model is None:
        reload_default_model()
    return _loaded_model
