from typing import List

import torch
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from hw_ss.base.base_metric import BaseMetric

class PESQMetric(BaseMetric):
    def __init__(self, fs=16000, mode="wb", *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.metric = PerceptualEvaluationSpeechQuality(fs, mode)

    def __call__(self, predict_wavs: torch.Tensor, audio_target: torch.Tensor, **kwargs):
        assert predict_wavs.shape == audio_target.shape, "Targets and predictions must be the same size"

        metrics = []
        for pred, target in zip(predict_wavs, audio_target):
            metrics.append(self.metric(pred.detach().cpu(), target.detach().cpu()).item())
        #metrics = 10 * torch.log10(torch.norm(audio_target, dim=1) / torch.norm(predict_wavs - audio_target, dim=1))
        return sum(metrics) / len(metrics)
