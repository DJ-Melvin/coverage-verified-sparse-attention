"""Load HuggingFace model and extract layer-level attention components.

Provides a thin wrapper that exposes Q/K/V/O projections, layer norms,
embedding, and LM head — everything needed for a standalone decode loop
without any serving framework.
"""

import torch
from dataclasses import dataclass
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


@dataclass
class ModelConfig:
    num_layers: int
    num_kv_heads: int
    num_q_heads: int
    num_q_per_kv: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    rope_theta: float
    rms_norm_eps: float
    max_position_embeddings: int
    model_name: str


class ModelForDecode:
    """Wraps a HuggingFace causal LM for layer-by-layer decode."""

    def __init__(self, model_name: str, dtype: torch.dtype = torch.bfloat16,
                 device: str = "cuda", load_in_4bit: bool = False,
                 device_map: Optional[str] = None):
        # device_map (e.g. "auto"/"balanced") shards the model's layers across
        # multiple GPUs for large-KV configurations (needed for Llama/Mistral
        # B=8 at >=65K). When set, self.device points at the input/embedding
        # device (cuda:0 under "auto"); the caller is responsible for moving
        # activations across per-layer devices during a manual decode loop.
        # Default None preserves the original single-device load exactly.
        self.device = torch.device("cuda:0" if device_map else device)
        self.dtype = dtype

        # Load config first to extract architecture info
        hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.config = ModelConfig(
            num_layers=hf_config.num_hidden_layers,
            num_kv_heads=getattr(hf_config, "num_key_value_heads",
                                 hf_config.num_attention_heads),
            num_q_heads=hf_config.num_attention_heads,
            num_q_per_kv=hf_config.num_attention_heads // getattr(
                hf_config, "num_key_value_heads", hf_config.num_attention_heads),
            head_dim=getattr(hf_config, "head_dim", None)
                     or hf_config.hidden_size // hf_config.num_attention_heads,
            hidden_size=hf_config.hidden_size,
            vocab_size=hf_config.vocab_size,
            rope_theta=getattr(hf_config, "rope_theta", 10000.0),
            rms_norm_eps=getattr(hf_config, "rms_norm_eps", 1e-5),
            max_position_embeddings=getattr(hf_config, "max_position_embeddings", 131072),
            model_name=model_name,
        )

        # Load model (optionally quantized)
        load_kwargs = dict(trust_remote_code=True)
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["device_map"] = {"": torch.device(device)}
        else:
            load_kwargs["dtype"] = dtype
            load_kwargs["device_map"] = device_map if device_map else device

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, **load_kwargs,
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_layers(self):
        """Return the list of transformer decoder layers."""
        return self.model.model.layers

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.model.embed_tokens(input_ids)

    def final_norm(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.model.model.norm(hidden)

    def lm_head(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.model.lm_head(hidden)

    def tokenize(self, text: str, max_length: Optional[int] = None) -> torch.Tensor:
        kwargs = {"return_tensors": "pt", "truncation": max_length is not None}
        if max_length is not None:
            kwargs["max_length"] = max_length
        return self.tokenizer(text, **kwargs).input_ids.to(self.device)
