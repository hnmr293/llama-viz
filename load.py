import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizer

assert transformers.__version__ >= "4.34.1"

def load_model(repo_id: str, model_args: dict, tokenizer_args: dict) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    model_args["cache_dir"] = "./models"
    tokenizer_args["cache_dir"] = "./models"

    model = AutoModelForCausalLM.from_pretrained(repo_id, **model_args)
    tokenizer = AutoTokenizer.from_pretrained(repo_id, **tokenizer_args)
    return model, tokenizer
