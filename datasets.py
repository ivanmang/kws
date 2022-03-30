from enum import Enum
import hashlib
import os
import random
import re
from manage_audio import AudioPreprocessor
from chainmap import ChainMap
import librosa
import numpy as np
import torch
import torch.utils.data as data
import pandas as pd


class SimpleCache(dict):
    def __init__(self, limit):
        super().__init__()
        self.limit = limit
        self.n_keys = 0

    def __setitem__(self, key, value):
        if key in self.keys():
            super().__setitem__(key, value)
        elif self.n_keys < self.limit:
            self.n_keys += 1
            super().__setitem__(key, value)
        return value


class DatasetType(Enum):
    TRAIN = 0
    DEV = 1
    TEST = 2


class SpeechDataset(data.Dataset):
    LABEL_SILENCE = "__silence__"
    LABEL_UNKNOWN = "__unknown__"
    df_list = []
    # audio_list = ['yes', "no", 'left', 'right', 'up', 'down', "on","off", "stop", "go"]
    audio_list = ['eight', 'sheila', 'nine', 'yes', 'one', 'no', 'left', 'tree', 'bed', 'bird', 'go', 'wow', 'seven',
                  'marvin', 'dog', 'three', 'two', 'house', 'backward', 'down', 'six', 'five', 'off', 'right', 'cat',
                  'zero', 'four', 'forward', 'stop', 'up', 'visual', 'learn', 'on', 'happy', 'follow']

    def __init__(self, _data, set_type, config):
        super().__init__()
        self.audio_files = list(_data.keys())
        self.set_type = set_type
        self.audio_labels = list(_data.values())
        config["bg_noise_files"] = list(filter(lambda x: x.endswith("wav"), config.get("bg_noise_files", [])))
        self.bg_noise_audio = [librosa.core.load(file, sr=8000)[0] for file in config["bg_noise_files"]]
        self.unknown_prob = config["unknown_prob"]
        self.silence_prob = config["silence_prob"]
        self.noise_prob = config["noise_prob"]
        self.input_length = config["input_length"]
        self.timeshift_ms = config["timeshift_ms"]
        self._audio_cache = SimpleCache(config["cache_size"])
        self._file_cache = SimpleCache(config["cache_size"])
        n_unk = len(list(filter(lambda x: x == 1, self.audio_labels)))
        self.n_silence = int(self.silence_prob * (len(self.audio_labels) - n_unk))
        self.audio_processor = AudioPreprocessor(n_mels=config["n_mels"], n_dct_filters=config["n_dct_filters"],
                                                 hop_ms=30)
        self.audio_preprocess_type = config["audio_preprocess_type"]

    @staticmethod
    def default_config():
        config = {"group_speakers_by_id": True, "silence_prob": 0.1, "noise_prob": 0.8, "n_dct_filters": 32,
                  "input_length": 8000, "n_mels": 32, "timeshift_ms": 100, "unknown_prob": 0.2, "train_pct": 85,
                  "dev_pct": 0.1, "test_pct": 0.1,
                  "wanted_words": ["one", "two", "three", "four", "five"],
                  "data_folder": "/data/DeepLearning/speech_commands_v0.02", "audio_preprocess_type": "MFCCs_delta"}
        return config

    def collate_fn(self, _data):
        x = None
        y = []
        for audio_data, label in _data:
            if self.audio_preprocess_type == "MFCCs_delta":
                audio_data = audio_data[35 * self.input_length // 1000:-35 * self.input_length // 1000]
                mfcc, d1, d2=self.audio_processor.compute_mfccs(audio_data)
                mfcc, d1, d2=mfcc.reshape(1, -1, 32), d1.reshape(1, -1, 32), d2.reshape(1, -1, 32)
                audio_tensor=torch.cat((torch.from_numpy(mfcc), torch.from_numpy(d1), torch.from_numpy(d2)), 0)
                audio_tensor=audio_tensor.unsqueeze(0)
                x=audio_tensor if x is None else torch.cat((x, audio_tensor), 0)


            if self.audio_preprocess_type == "MFCCs":
                # audio_tensor = torch.from_numpy(self.audio_processor.compute_mfccs(audio_data))
                # audio_tensor=torch.unsqueeze(audio_tensor, 0)
                # 20200629 clip data to get core 930ms, use 30ms hop length
                audio_data = audio_data[35 * self.input_length // 1000:-35 * self.input_length // 1000]
                audio_tensor = torch.from_numpy(
                    self.audio_processor.compute_mfccs(audio_data).reshape(1, -1, 32))
                x = audio_tensor if x is None else torch.cat((x, audio_tensor), 0)

            y.append(label)

        return x, torch.tensor(y)

    def load_csv(self):

        for keyword in self.audio_list:  # keylist: #["eight"]:#config["wanted_words"]:

            self.df_list.append(pd.read_csv("/data/DeepLearning/speech_commands_v0.02_csv/" + keyword + ".csv"))
        print("End load audio csv")

    def _timeshift_audio(self, _data):
        shift = (8000 * self.timeshift_ms) // 1000
        shift = random.randint(-shift, shift)
        a = -min(0, shift)
        b = max(0, shift)
        _data = np.pad(_data, (a, b), "constant")
        return _data[:len(_data) - a] if a else _data[b:]

    def load_audio(self, example, silence=False):

        # print(f"load audio {example}")
        if silence:
            example = "__silence__"
        if random.random() < 0.7 or not self.set_type == DatasetType.TRAIN:
            try:
                return self._audio_cache[example]
            except KeyError:
                pass
        in_len = self.input_length
        if self.bg_noise_audio:
            bg_noise = random.choice(self.bg_noise_audio)
            a = random.randint(0, len(bg_noise) - in_len - 1)
            bg_noise = bg_noise[a:a + in_len]
        else:
            bg_noise = np.zeros(in_len)

        if silence:
            _data = np.zeros(in_len, dtype=np.float32)
        else:
            file_data = self._file_cache.get(example)
            example_split = example.split("/")
            print(example_split)

            c_df = self.df_list[self.audio_list.index(example_split[4])]

            # _data = librosa.core.load(example, sr=8000)[0] if file_datais None else file_data
            tmp_data = (c_df.loc[c_df['1'] == example_split[5]].values[0][2:]).astype(float)

            tmp_data = tmp_data[~np.isnan(tmp_data)]
            _data = tmp_data if file_data is None else file_data
            self._file_cache[example] = _data
        _data = np.pad(_data, (0, max(0, in_len - len(_data))), "constant")
        if self.set_type == DatasetType.TRAIN:
            _data = self._timeshift_audio(_data)

        if random.random() < self.noise_prob or silence:
            a = random.random() * 0.1
            _data = np.clip(a * bg_noise + _data, -1, 1)

        self._audio_cache[example] = _data
        return _data

    @classmethod
    def splits(cls, config):
        folder = config["data_folder"]
        wanted_words = config["wanted_words"]
        unknown_prob = config["unknown_prob"]
        train_pct = config["train_pct"]
        dev_pct = config["dev_pct"]
        test_pct = config["test_pct"]

        words = {word: i + 2 for i, word in enumerate(wanted_words)}
        words.update({cls.LABEL_SILENCE: 0, cls.LABEL_UNKNOWN: 1})
        sets = [{}, {}, {}]
        unknowns = [0] * 3
        bg_noise_files = []
        unknown_files = []

        for folder_name in os.listdir(folder):
            path_name = os.path.join(folder, folder_name)
            is_bg_noise = False
            if os.path.isfile(path_name):
                continue
            if folder_name in words:
                label = words[folder_name]
            elif folder_name == "_background_noise_":
                is_bg_noise = True
            else:
                label = words[cls.LABEL_UNKNOWN]

            for filename in os.listdir(path_name):
                wav_name = os.path.join(path_name, filename)
                if is_bg_noise and os.path.isfile(wav_name):
                    bg_noise_files.append(wav_name)
                    continue
                elif label == words[cls.LABEL_UNKNOWN]:
                    unknown_files.append(wav_name)
                    continue
                if config["group_speakers_by_id"]:
                    hashname = re.sub(r"_nohash_.*$", "", filename)
                max_no_wavs = 2 ** 27 - 1
                bucket = int(hashlib.sha1(hashname.encode()).hexdigest(), 16)
                bucket = (bucket % (max_no_wavs + 1)) * (100. / max_no_wavs)
                if bucket < dev_pct:
                    tag = DatasetType.DEV
                elif bucket < test_pct + dev_pct:
                    tag = DatasetType.TEST
                else:
                    tag = DatasetType.TRAIN
                sets[tag.value][wav_name] = label

        for tag in range(len(sets)):
            unknowns[tag] = int(unknown_prob * len(sets[tag]))
        random.shuffle(unknown_files)
        a = 0
        for i, dataset in enumerate(sets):
            b = a + unknowns[i]
            unk_dict = {u: words[cls.LABEL_UNKNOWN] for u in unknown_files[a:b]}
            dataset.update(unk_dict)
            a = b

        train_cfg = ChainMap(dict(bg_noise_files=bg_noise_files), config)
        test_cfg = ChainMap(dict(bg_noise_files=bg_noise_files, noise_prob=0), config)
        datasets = (cls(sets[0], DatasetType.TRAIN, train_cfg), cls(sets[1], DatasetType.DEV, test_cfg),
                    cls(sets[2], DatasetType.TEST, test_cfg))
        with open('data/test.csv', 'w') as f:
            for key in sets[0].keys():
                f.write("%s,%s\n" % (key, sets[0][key]))

        return datasets

    def __getitem__(self, index):
        if index >= len(self.audio_labels):
            return self.load_audio(None, silence=True), 0
        return self.load_audio(self.audio_files[index]), self.audio_labels[index]

    def __len__(self):
        return len(self.audio_labels) + self.n_silence
