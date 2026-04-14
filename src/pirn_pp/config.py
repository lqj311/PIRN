from dataclasses import dataclass


@dataclass(slots=True)
class PIRNPPConfig:
    """Configuration for PIRN++ modules."""

    dim: int = 256
    num_proto_rgb: int = 32
    num_proto_xyz: int = 32
    sinkhorn_iters: int = 6
    sinkhorn_tau: float = 0.08
    assign_mass_floor: float = 0.05
    ema_momentum: float = 0.999
    update_uncertainty_threshold: float = 0.55
    mnc_steps: int = 2
    ib_beta: float = 0.03
    proto_align_weight: float = 0.10
    consistency_weight: float = 0.20
    entropy_reg_weight: float = 0.02

