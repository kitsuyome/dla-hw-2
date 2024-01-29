import torch
from torch import Tensor


class SISDRLoss(torch.nn.Module):
    def __init__(self, alpha=0.1, beta=0.1, eps=1e-9):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.eps = eps
    def forward(self, ests, audio_target,
                **batch) -> Tensor:
        audio_target = audio_target.unsqueeze(1)
        loss = 0.0
        for i, est in enumerate(ests):
            gamma = torch.bmm(audio_target, est.transpose(1, 2))
            gamma /= torch.bmm(audio_target, audio_target.transpose(1, 2))
            in_norm = torch.bmm(gamma, audio_target).squeeze(1)
            up = torch.linalg.norm(in_norm, ord=2, dim=1)
            down = torch.linalg.norm(in_norm - est.squeeze(1), ord=2, dim=1)

            result = 20 * torch.log10(up / (down + self.eps) + self.eps)
            if i == 0:
                loss -= (1 - self.alpha - self.beta) * result.sum()
            elif i == 1:
                loss -= self.alpha * result.sum()
            elif i == 2:
                loss -= self.beta * result.sum()
        return loss
