import logging
from pathlib import Path
from glob import glob
import os

from hw_ss.base.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class CustomDirAudioDataset(BaseDataset):
    def __init__(self, data_dir, *args, **kwargs):
        mixed, targets, refs = data_dir + "/mix/*", data_dir + "/targets/*", data_dir + "/refs/*"
        data = dict()
        for path in glob(mixed):
            filename = os.path.basename(path).split('-')[0]
            if filename not in data and ".wav" in path:
                data[filename] = [path]

        for path in glob(targets):
            filename = os.path.basename(path).split('-')[0]
            if ".wav" in path:
                data[filename].append(path)

        for path in glob(refs):
            filename = os.path.basename(path).split('-')[0]
            if ".wav" in path:
                data[filename].append(path)
        new_data = []
        for key, value in data.items():
            if len(value) == 3:
                new_data.append({
                    "path": value
                })

        super().__init__(new_data, *args, **kwargs)

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        audio_paths = data_dict["path"]
        for x in audio_paths:
            if '-mixed' in x:
                audio_path_mixed = x
            if '-target' in x:
                audio_path_target = x
            if '-ref' in x:
                audio_path_ref = x
        audio_wave_target, audio_wave_mixed, audio_wave_ref = self.load_audio(audio_path_target), self.load_audio(audio_path_mixed), self.load_audio(audio_path_ref)
        audio_wave_target, audio_spec_target = self.process_wave(audio_wave_target)
        audio_wave_mixed, audio_spec_mixed = self.process_wave(audio_wave_mixed)
        audio_wave_ref, audio_spec_ref = self.process_wave(audio_wave_ref)
        return {
            "audio_target": audio_wave_target,
            "audio_mixed": audio_wave_mixed,
            "audio_ref": audio_wave_ref,
            "spectrogram_target": audio_spec_target,
            "spectrogram_mixed": audio_spec_mixed,
            "spectrogram_ref": audio_spec_ref,
            "duration": audio_wave_target.size(1) / self.config_parser["preprocessing"]["sr"],
            "audio_path": audio_path_target,
        }
    @staticmethod
    def _sort_index(index):
        return index

    @staticmethod
    def _assert_index_is_valid(index):
        pass