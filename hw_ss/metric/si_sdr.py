from typing import List
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
import torch

from hw_ss.base.base_metric import BaseMetric

class SISDRMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.metric = ScaleInvariantSignalNoiseRatio()

    def __call__(self, predict_wavs: torch.Tensor, audio_target: torch.Tensor, **kwargs):
        assert predict_wavs.shape == audio_target.shape, "Targets and predictions must be the same size"

        metrics = []
        for pred, target in zip(predict_wavs, audio_target):
            metrics.append(self.metric(pred.detach().cpu(), target.detach().cpu()).item())
        #metrics = 10 * torch.log10(torch.norm(audio_target, dim=1) / torch.norm(predict_wavs - audio_target, dim=1))
        return sum(metrics) / len(metrics)
