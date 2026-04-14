import torch

from pirn_pp import PIRNPPConfig, PIRNPlusPlus


def main() -> None:
    cfg = PIRNPPConfig(dim=256, num_proto_rgb=32, num_proto_xyz=32)
    model = PIRNPlusPlus(cfg)

    f_rgb = torch.randn(2, 196, 256)
    f_xyz = torch.randn(2, 196, 256)
    target = torch.randint(0, 2, (2, 196)).float()

    out = model(f_rgb, f_xyz, update_teacher=True)
    losses = model.compute_loss(out, target)
    losses["total"].backward()
    print({k: float(v.detach()) for k, v in losses.items()})


if __name__ == "__main__":
    main()

