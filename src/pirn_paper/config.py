from dataclasses import dataclass


@dataclass(slots=True)
class PIRNConfig:
    # Feature shape
    dim: int = 768
    num_tokens: int = 196

    # Prototype bank size
    num_proto_rgb: int = 24
    num_proto_sn: int = 24

    # BPA
    sinkhorn_tau: float = 0.07
    sinkhorn_iters: int = 7

    # APR
    apr_eps: float = 1e-6

    # MNC
    mnc_heads: int = 8
    mnc_dropout: float = 0.1

    # Loss weights
    rec_weight: float = 1.0
    sem_weight: float = 0.1
    div_weight: float = 0.05

