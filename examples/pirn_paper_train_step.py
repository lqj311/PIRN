import torch

from pirn_paper import PIRNConfig, PIRNModel


def main() -> None:
    """
    Single train step for PIRN paper pipeline.
    In real experiments, replace f_rgb/f_sn with backbone features
    extracted from RGB and surface-normal (or depth) inputs.
    """
    cfg = PIRNConfig(
        dim=768,
        num_tokens=196,
        num_proto_rgb=24,
        num_proto_sn=24,
        sinkhorn_tau=0.07,
        sinkhorn_iters=7,
    )
    model = PIRNModel(cfg)
    model.train()

    # Placeholder features; replace with DINOv2/SN encoder outputs.
    f_rgb = torch.randn(2, 196, 768)
    f_sn = torch.randn(2, 196, 768)

    out = model(f_rgb, f_sn, update_proto=True)
    losses = model.compute_loss(out, f_rgb, f_sn)
    losses["total"].backward()

    print({k: float(v.detach()) for k, v in losses.items()})
    print("image_anomaly:", out["image_anomaly"].detach().tolist())


if __name__ == "__main__":
    main()

