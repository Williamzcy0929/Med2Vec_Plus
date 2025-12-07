import torch, torch.nn as nn

class AspectFusion(nn.Module):
    def __init__(self, d: int, d_demo: int = 0, d_fuse: int = 256):
        super().__init__()
        self.proj_demo = nn.Linear(d_demo, d) if d_demo > 0 else None
        self.gate = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 1))
        self.out = nn.Sequential(nn.Linear(d + (d if d_demo > 0 else 0), d_fuse), nn.ReLU(), nn.LayerNorm(d_fuse))

    def forward(self, z_dx: torch.Tensor, z_proc: torch.Tensor, z_treat: torch.Tensor, demo: torch.Tensor = None):
        z_stack = torch.stack([z_dx, z_proc, z_treat], dim=2)
        q = z_stack.mean(dim=2)
        alphas = []
        for i in range(3):
            a = torch.sigmoid(self.gate(torch.cat([q, z_stack[:, :, i, :]], dim=-1)))
            alphas.append(a)
        alphas = torch.stack(alphas, dim=2)
        zmix = (alphas * z_stack).sum(dim=2)
        if self.proj_demo is not None and demo is not None:
            gd = self.proj_demo(demo)
            fused = torch.cat([zmix, gd], dim=-1)
        else:
            fused = zmix
        return self.out(fused)
