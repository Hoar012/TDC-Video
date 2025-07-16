# Copyright (2024) Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from PIL import Image
from typing import List
import torch
from transformers import WhisperFeatureExtractor
from transformers.models.llava import LlavaProcessor
import soundfile as sf
import librosa
import re
import numpy as np
from torch.nn.utils.rnn import pad_sequence
# from .utils import sample_image, sample_video


class AudioProcessor:
    def __init__(self, whisper_path):
        self.wav_processor = WhisperFeatureExtractor.from_pretrained(whisper_path)
        self.target_sample_rate = 16000

    def read_audio(self, audio_path, start_time=None, end_time=None):
        audio, sr = sf.read(audio_path)
        if len(audio.shape) > 1: # stereo to mono
            audio = audio[:, 0]
        if start_time is not None or end_time is not None:
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            audio = audio[start_sample:end_sample]
        return audio, sr
    
    def load_audio_file(self, audio_path, start_time=None, end_time=None):
        if isinstance(audio_path, str):
            audio_path = [audio_path]
            start_time = [start_time]
            end_time = [end_time]
        else:
            if start_time is None:
                start_time = [None] * len(audio_path)
            if end_time is None:
                end_time = [None] * len(audio_path)
        audio, sr = self.read_audio(audio_path[0], start_time[0], end_time[0])
        for idx in range(1, len(audio_path)):
            expand_audio, expand_sr = self.read_audio(audio_path[idx], start_time[idx], end_time[idx])
            assert sr==expand_sr, "audio sample rate is different!"
            sil = np.zeros(sr, dtype=float)
            audio = np.concatenate((audio, sil, expand_audio), axis=0)
        if len(audio) < sr:
            sil = np.zeros(sr - len(audio), dtype=float)
            audio = np.concatenate((audio, sil), axis=0)
        if sr != self.target_sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sample_rate)
            sr = self.target_sample_rate
        audio_spectrogram = self.wav_processor(audio, sampling_rate=sr, return_tensors="pt")["input_features"].squeeze()
        return {
            "audio_spectrogram": audio_spectrogram,
            "audio_wav": audio,
        }
    
    def batch_collate(self, samples):
        audio_spectrogram = [s["audio_spectrogram"] for s in samples]
        audio_spectrogram = torch.stack(audio_spectrogram, dim=0)
        audio_wav = [torch.from_numpy(s["audio_wav"]) for s in samples]
        wav_length = torch.tensor([len(s["audio_wav"]) for s in samples])
        audio_wav = pad_sequence(audio_wav, batch_first=True, padding_value=0)
        audio_wav_mask = torch.arange(audio_wav.size(1)).unsqueeze(0) >= wav_length.unsqueeze(1)
        return {
            "audio_spectrogram": audio_spectrogram.half(),
            "audio_wav": audio_wav.half(),
            "audio_wav_mask": audio_wav_mask.half(),
        }
    
    def dummy_audio_input(self):
        sr = self.target_sample_rate
        audio = np.zeros(sr, dtype=float)
        audio_spectrogram = self.wav_processor(audio, sampling_rate=sr, return_tensors="pt")["input_features"].squeeze()
        return self.batch_collate([{
            "audio_spectrogram": audio_spectrogram,
            "audio_wav": audio,
        }])


class Processor(object):
    def __init__(
            self,
            whisper_model_path
        ):
        self.audio_processor = AudioProcessor(whisper_path=whisper_model_path)

    def get_text_inputs(self, text):
        prompt_ids = self.tokenizer.encode(text, add_special_tokens=True)
        prompt_ids = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(dim=0)
        return prompt_ids

    def __call__(self, data):
        # load audio
        audios = []
        if "audio" in data:
            for audio in data["audio"]:
                audios.append(self.audio_processor.load_audio_file(
                    audio_path=audio['audio_file'],
                    start_time=audio['start_time'],
                    end_time=audio['end_time'],
                ))

        inputs = dict()
        inputs.update(self.audio_processor.batch_collate(audios))
        return inputs