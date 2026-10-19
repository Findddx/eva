"""Map the checkpoint config.json onto the field names used by the official model."""
import json
from dataclasses import dataclass


@dataclass
class ModelArgs:
    max_batch_size: int = 1
    max_seq_len: int = 2048
    temperature: float = 0.0
    vocab_size: int = 129280
    dim: int = 5120
    moe_inter_dim: int = 2304
    n_layers: int = 40
    n_mtp_layers: int = 3
    n_heads: int = 64
    n_routed_experts: int = 384
    n_shared_experts: int = 1
    n_activated_experts: int = 6
    score_func: str = "sqrtsoftplus"
    gate_temp: float = 1.0
    norm_topk_prob: bool = True
    route_scale: float = 1.5
    swiglu_limit: float = 10.0
    q_lora_rank: int = 1280
    head_dim: int = 512
    rope_head_dim: int = 64
    norm_eps: float = 1e-20
    o_groups: int = 8
    o_lora_rank: int = 1024
    window_size: int = 128
    compress_ratios: tuple = ()
    kv_source_layers: tuple = ()
    index_source_layers: tuple = ()
    compress_rope_theta: float = 160000.0
    original_seq_len: int = 65536
    rope_theta: float = 10000.0
    rope_factor: float = 16.0
    beta_fast: int = 32
    beta_slow: int = 1
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 512
    candidate_source_layer: int = 20
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    engram_layer_ids: tuple = ()
    engram_num_embeddings: tuple = ()
    engram_max_ngram_size: int = 4
    engram_vocab_size: int = 16000000
    engram_n_heads: int = 8
    engram_head_dim: int = 256
    engram_pad_id: int = 2
    engram_compressed_vocab_size: int = 99092
    dspark_block_size: int = 5
    dspark_noise_token_id: int = 128799
    dspark_target_layer_ids: tuple = ()
    dspark_markov_rank: int = 256
    dspark_n_routed_experts: int = 128
    dspark_n_activated_experts: int = 3

    @staticmethod
    def from_config(path: str) -> "ModelArgs":
        raw = json.load(open(path, encoding="utf-8"))
        text = raw["text_config"]
        rope = text.get("rope_scaling") or {}
        ratios = text["compress_ratios"]
        # The published list has one entry per backbone layer plus the MTP blocks.
        return ModelArgs(
            vocab_size=text["vocab_size"],
            dim=text["hidden_size"],
            moe_inter_dim=text["moe_intermediate_size"],
            n_layers=text["num_hidden_layers"],
            n_mtp_layers=text.get("num_nextn_predict_layers", 0),
            n_heads=text["num_attention_heads"],
            n_routed_experts=text["n_routed_experts"],
            n_shared_experts=text["n_shared_experts"],
            n_activated_experts=text["num_experts_per_tok"],
            score_func=text["scoring_func"],
            norm_topk_prob=text["norm_topk_prob"],
            route_scale=text["routed_scaling_factor"],
            swiglu_limit=text.get("swiglu_limit", 0.0),
            q_lora_rank=text["q_lora_rank"],
            head_dim=text["head_dim"],
            rope_head_dim=text["qk_rope_head_dim"],
            norm_eps=text["rms_norm_eps"],
            o_groups=text["o_groups"],
            o_lora_rank=text["o_lora_rank"],
            window_size=text["sliding_window"],
            compress_ratios=tuple(ratios[: text["num_hidden_layers"]]),
            kv_source_layers=tuple(text["kv_source_layer_ids"]),
            index_source_layers=tuple(text["index_source_layer_ids"]),
            compress_rope_theta=text["compress_rope_theta"],
            original_seq_len=rope.get("original_max_position_embeddings", 0),
            rope_theta=text["rope_theta"],
            rope_factor=rope.get("factor", 1.0),
            beta_fast=rope.get("beta_fast", 32),
            beta_slow=rope.get("beta_slow", 1),
            index_n_heads=text["index_n_heads"],
            index_head_dim=text["index_head_dim"],
            index_topk=text["index_topk"],
            candidate_source_layer=text["candidate_source_layer_id"],
            candidate_topk_blocks=text["candidate_topk_blocks"],
            candidate_block_size=text["candidate_block_size"],
            hc_mult=text["hc_mult"],
            hc_sinkhorn_iters=text["hc_sinkhorn_iters"],
            hc_eps=text["hc_eps"],
            engram_layer_ids=tuple(text["engram_layer_ids"]),
            engram_num_embeddings=tuple(text["engram_num_embeddings"]),
            engram_max_ngram_size=text["engram_max_ngram_size"],
            engram_vocab_size=text["engram_vocab_size"],
            engram_n_heads=text["engram_n_heads"],
            engram_head_dim=text["engram_head_dim"],
            engram_pad_id=text["engram_pad_token_id"],
            engram_compressed_vocab_size=text["engram_compressed_vocab_size"],
            dspark_block_size=text.get("dspark_block_size", 0),
            dspark_noise_token_id=text.get("dspark_noise_token_id", 0),
            dspark_target_layer_ids=tuple(text.get("dspark_target_layer_ids", ())),
            dspark_markov_rank=text.get("dspark_markov_rank", 256),
            dspark_n_routed_experts=text.get("dspark_n_routed_experts", 0),
            dspark_n_activated_experts=text.get("dspark_num_experts_per_tok", 0),
        )
