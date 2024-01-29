import json
import logging
import os
import shutil
from glob import glob
from pathlib import Path

import torchaudio
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm

from hw_ss.base.base_dataset import BaseDataset
from hw_ss.utils import ROOT_PATH
from hw_ss.mixer.mixer import LibriSpeechSpeakerFiles, MixtureGenerator
from hw_ss.datasets.librispeech_dataset import LibrispeechDataset


logger = logging.getLogger(__name__)

class MixerDataset(BaseDataset):
    def __init__(self, part, data_dir=None, index_dir=None, train=True, *args, **kwargs):
        self.train = train

        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "mix"
            data_dir.mkdir(exist_ok=True, parents=True)
        else:
            data_dir = Path(data_dir)
        self._data_dir = data_dir

        if index_dir is None:
            index_dir = ROOT_PATH / "data" / "mix"
            index_dir.mkdir(exist_ok=True, parents=True)
        else:
            index_dir = Path(index_dir)
        self._index_dir = index_dir

        index = self._get_or_load_index(part)

        super().__init__(index, *args, **kwargs)

    def _load_part(self, part):
        librispeech_data_dir = ROOT_PATH / "data" / "datasets" / "librispeech"
        path = librispeech_data_dir / part
        if not path.exists():
            LibrispeechDataset(part, librispeech_data_dir)
        speakers = [el.name for el in os.scandir(path)]
        speakers_files = [LibriSpeechSpeakerFiles(i, path, audioTemplate='*.flac') for i in speakers]
        mixer = MixtureGenerator(speakers_files, self._data_dir, test=not(self.train))
        params = {
            "snr_levels": [-5, 5],
            "num_workers": 2,
            "update_steps": 100,
            "trim_db":None,
            "vad_db": 20
        }
        if (self.train):
            params["audioLen"] = 3
            params["trim_db"] = 20

        mixer.generate_mixes(**params)

    def _get_or_load_index(self, part):
        index_path = self._index_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part):
        index = []
        split_dir = self._data_dir / part

        if not split_dir.exists():
            self._load_part(part)

        wavs = dict()
        for path in glob(str(split_dir) + "/*"):
            if path.endswith(".wav"):
                filename = os.path.basename(path)
                speaker, type = filename.split('-')
                if (not wavs.get(speaker)):
                    wavs[speaker] = []
                wavs[speaker].append(type)
        for wav_path, types in tqdm(
               wavs.items(), desc=f"Preparing index: {part}"
        ):
            if (len(types) == 3):
                full_wav_name = wav_path + "-target.wav"
                path = split_dir / full_wav_name
                t_info = torchaudio.info(str(path))
                length = t_info.num_frames / t_info.sample_rate
                index.append(
                    {
                        "path": str(split_dir / wav_path),
                        "audio_len": length
                    }
                )
        return index
    def __getitem__(self, ind):
        data_dict = self._index[ind]
        audio_path = data_dict["path"]
        audio_wave_target, audio_wave_mixed, audio_wave_ref = self.load_audio(audio_path + '-target.wav'), self.load_audio(audio_path + '-mixed.wav'), self.load_audio(audio_path + '-ref.wav')
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
            "audio_path": audio_path,
        }