import logging
import torch
from typing import List
import os

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    num_items = len(dataset_items)
    max_audio_length = {
        "target": max([item['audio_target'].shape[1] for item in dataset_items]),
        "mixed": max([item['audio_mixed'].shape[1] for item in dataset_items]),
        "ref": max([item['audio_ref'].shape[1] for item in dataset_items])
    }
    max_spec_length = {
        "target": max([item['spectrogram_target'].shape[2] for item in dataset_items]),
        "mixed": max([item['spectrogram_mixed'].shape[2] for item in dataset_items]),
        "ref": max([item['spectrogram_ref'].shape[2] for item in dataset_items])
    }

    audio = {
        "target": torch.zeros(num_items, max_audio_length["target"]),
        "mixed": torch.zeros(num_items, max_audio_length["mixed"]),
        "ref": torch.zeros(num_items, max_audio_length["ref"]),
    }
    spectrogram = {
        "target": torch.zeros(num_items, dataset_items[0]['spectrogram_target'].shape[1], max_spec_length["target"]),
        "mixed": torch.zeros(num_items, dataset_items[0]['spectrogram_mixed'].shape[1], max_spec_length["mixed"]),
        "ref": torch.zeros(num_items, dataset_items[0]['spectrogram_ref'].shape[1], max_spec_length["ref"]),
    }
    speaker_id = torch.zeros(num_items).long()
    audio_path = []
    for i, item in enumerate(dataset_items):
        for key in audio.keys():
            audio[key][i, :item['audio_' + key].shape[1]] = item['audio_' + key].squeeze(0)
            spectrogram[key][i, :, :item['spectrogram_' + key].shape[2]] = item['spectrogram_' + key].squeeze(0)
            path = os.path.basename(item['audio_path'])
            audio_path.append(path)
            id = path.split('_')[0]
            speaker_id[i] = int(id)

    result_batch = {
        "audio_target": audio["target"],
        "audio_mixed": audio["mixed"],
        "audio_ref": audio["ref"],
        "spectrogram_target": spectrogram["target"],
        "spectrogram_mixed": spectrogram["mixed"],
        "spectrogram_ref": spectrogram["ref"],
        "speaker_id": speaker_id,
        "audio_path": audio_path
    }
    return result_batch