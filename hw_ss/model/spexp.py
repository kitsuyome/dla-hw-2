from torch import nn
from torch.nn import Sequential
import torch

from hw_ss.base import BaseModel

class TCNBlock(nn.Module):
    def __init__(self, O, P, Q, B, embed_dim=None):
        super().__init__()

        if (embed_dim):
            self.conv = Sequential(
                nn.Conv1d(O+embed_dim, P, 1),
                nn.PReLU(),
                LayerNorm(P), # must be Global Layer Normalization
                nn.Conv1d(P, P, Q, dilation=B, padding= (B * (Q - 1) + 1) // 2),
                nn.PReLU(),
                LayerNorm(P), # must be Global Layer Normalization
                nn.Conv1d(P, O, 1),
            )
        else:
            self.conv = Sequential(
                nn.Conv1d(O, P, 1),
                nn.PReLU(),
                LayerNorm(P), # must be Global Layer Normalization
                nn.Conv1d(P, P, Q, dilation=B, padding= (B * (Q - 1) + 1) // 2),
                nn.PReLU(),
                LayerNorm(P), # must be Global Layer Normalization
                nn.Conv1d(P, O, 1),
            )
    def forward(self, inputs, v=None):
        outputs = inputs
        if (v is not None):
            v = v.unsqueeze(2)
            v = v.repeat(1, 1, inputs.shape[2])
            outputs = torch.concat([outputs, v], dim=1)
        outputs = self.conv(outputs)
        outputs = outputs[:,:,:inputs.shape[2]]
        outputs += inputs
        return outputs

class StackedTCN(nn.Module):
    def __init__(self, O, P, Q, B, embed_dim=256):
        super().__init__()

        self.tcn = nn.ModuleList()
        while (B > 0):
            self.tcn.append(TCNBlock(O, P, Q, B, embed_dim=embed_dim))
            embed_dim=None
            B -= 1

    def forward(self, inputs, v):
        output = inputs
        for layer in self.tcn:
            output = layer(output, v)
            v = None
        return output

class ResNetBlock(nn.Module):
    def __init__(self, O, P):
        super().__init__()

        self.conv = Sequential(
            nn.Conv1d(O, P, 1),
            nn.BatchNorm1d(P),
            nn.PReLU(),
            nn.Conv1d(P, O, 1),
            nn.BatchNorm1d(O)
        )
        self.out = Sequential(
            nn.PReLU(),
            nn.MaxPool1d(3)
        )

    def forward(self, input):
        output = self.conv(input)
        output += input
        output = self.out(output)
        return output


class SpeechEncoder(nn.Module):
    def __init__(self, N, L1, L2, L3):
        super().__init__()

        self.short = nn.Conv1d(1, N, L1, stride=L1//2, padding=(L1 - 1)//2)
        self.middle = nn.Conv1d(1, N, L2, stride=L1//2, padding=(L2 - 1)//2)
        self.long = nn.Conv1d(1, N, L3, stride=L1//2, padding=(L3 - 1)//2)
        self.act = nn.ReLU()

    def forward(self, audio):
        audio = audio.unsqueeze(1)
        outputs = [self.act(self.short(audio)), self.act(self.middle(audio)), self.act(self.long(audio))]
        length = min([outputs[0].shape[-1], outputs[1].shape[-1], outputs[2].shape[-1]])
        outputs = outputs[0][:,:,:length], outputs[1][:,:,:length], outputs[2][:,:,:length]
        e1, e2, e3 = outputs
        outputs = torch.concat(outputs, dim=1)
        return outputs, (e1, e2, e3)

class LayerNorm(nn.LayerNorm):
    def __init__(self, N):
        super().__init__(N)

    def forward(self, inputs):
        inputs = inputs.transpose(1, 2)
        outputs = super().forward(inputs)
        outputs = outputs.transpose(1, 2)
        return outputs

class SpeakerEncoder(nn.Module):
    def __init__(self, N, O, P, num_rn=3, embed_dim=256, num_speakers=None):
        super().__init__()

        self.conv1 = Sequential(
            LayerNorm(3 * N),
            nn.Conv1d(3 * N, O, 1),
        )
        self.resnet = nn.ModuleList()
        for _ in range(num_rn):
            self.resnet.append(ResNetBlock(O, P))

        self.conv2 = Sequential(
            nn.Conv1d(O, embed_dim, 1)
        )
        if (num_speakers):
            self.head = Sequential(
                nn.Linear(embed_dim, num_speakers)
            )
    def forward(self, inputs):
        outputs = self.conv1(inputs)
        for layer in self.resnet:
            outputs = layer(outputs)
        outputs = self.conv2(outputs)
        outputs = outputs.mean(dim=2) # AvgPool
        classification = None
        if (hasattr(self, 'head')):
            classification = self.head(outputs)
        return outputs, classification

class SpeakerExtractor(nn.Module):
    def __init__(self, N, O, P, Q, embed_dim=256, B=8, R=4):
        super().__init__()

        self.N = N
        self.conv = Sequential(
            LayerNorm(3 * N),
            nn.Conv1d(3 * N, O, 1)
        )
        self.tcn = nn.ModuleList()
        for _ in range(R):
            self.tcn.append(StackedTCN(O, P, Q, B, embed_dim=embed_dim))

        self.short = Sequential(
            nn.Conv1d(O, 1, 1),
            nn.ReLU()
        )
        self.middle = Sequential(
            nn.Conv1d(O, 1, 1),
            nn.ReLU()
        )
        self.long = Sequential(
            nn.Conv1d(O, 1, 1),
            nn.ReLU()
        )

    def forward(self, inputs, v):
        length = inputs.shape[-1]
        outputs = self.conv(inputs)
        for layer in self.tcn:
            outputs = layer(outputs, v)
        return self.short(outputs)[:,:,:length], self.middle(outputs)[:,:,:length], self.long(outputs)[:,:,:length]


class SpeechDecoder(nn.Module):
    def __init__(self, N, L1, L2, L3):
        super().__init__()

        self.short = nn.ConvTranspose1d(N, 1, L1, stride=L1//2, padding=(L1 - 1) // 2, output_padding=(L1 - 1)//2 - 1)
        self.middle = nn.ConvTranspose1d(N, 1, L2, stride=L1//2, padding=(L2 - 1) // 2, output_padding=(L1 - 1)//2 - 1)
        self.long = nn.ConvTranspose1d(N, 1, L3, stride=L1//2, padding=(L3 - 1) // 2, output_padding=(L1 - 1)//2 - 1)

    def forward(self, m1, m2, m3, e1, e2, e3, length):
        e1, e2, e3 = m1 * e1, m2 * e2, m3 * e3
        e1, e2, e3 = self.short(e1), self.middle(e2), self.long(e3)
        e1, e2, e3 = e1[:,:,:length], e2[:,:,:length], e3[:,:,:length]
        return e1, e2, e3

class SpExp(BaseModel):
    def __init__(self, N, L1, L2, L3, O=256, P=512, Q=3, num_speakers=None, speaker_dim=256, **batch):
        super().__init__(**batch)

        self.speech_encoder = SpeechEncoder(N, L1, L2, L3)
        self.speaker_extractor = SpeakerExtractor(N, O, P, Q, embed_dim=speaker_dim)
        self.speech_decoder = SpeechDecoder(N, L1, L2, L3)

        self.speaker_encoder = SpeakerEncoder(N, O, P, num_speakers=num_speakers, embed_dim=speaker_dim)

    def forward(self, audio_mixed, audio_ref, audio_target, **batch):
        length = audio_target.shape[-1]

        speech, e = self.speech_encoder(audio_mixed)
        speaker, _ = self.speech_encoder(audio_ref)

        speaker, cls = self.speaker_encoder(speaker)

        m = self.speaker_extractor(speech, speaker)
        short, middle, long = self.speech_decoder(*m, *e, length)

        return (short, middle, long), cls

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
