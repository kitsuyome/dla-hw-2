import torch
from torch import Tensor
from hw_ss.loss.SISDRLoss import SISDRLoss
from torch.nn import CrossEntropyLoss

class MultiLoss(torch.nn.Module):
    def __init__(self, alpha=0.1, beta=0.1, gamma=0.1, eps=1e-9):
        super().__init__()

        self.sisdr = SISDRLoss(alpha, beta, eps)
        self.ce = CrossEntropyLoss()
        self.gamma = gamma
    def forward(self, ests, audio_target, logits, speaker_id,
                **batch) -> Tensor:
        sisdr = self.sisdr(ests, audio_target)
        if (self.training):
            ce = self.ce(logits, speaker_id)
            result = (1 - self.gamma) * sisdr + self.gamma * ce
        else:
            result = sisdr
        return result
