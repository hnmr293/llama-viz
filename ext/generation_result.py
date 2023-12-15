from dataclasses import dataclass
import torch

@dataclass
class GenerationResult:
    input: str
    input_ids: torch.Tensor
    input_tokens: list[str]
    output: str
    output_ids: torch.Tensor
    output_tokens: list[str]

    attentions: list
    all_hidden_states: tuple[tuple[torch.Tensor]]

    def __str__(self):
        return self.output

@dataclass
class GenerationInfo:
    device: torch.device
    seed: int
    time: int
    time_per_token: float

    def __str__(self):
        return f"""seed={self.seed:d}
device={self.device.type}:{self.device.index}
time={self.time/1000000:.1f}ms
     {self.time_per_token/1000000:.1f}ms/token ({1000000000/self.time_per_token:.1f}tokens/s)
"""
