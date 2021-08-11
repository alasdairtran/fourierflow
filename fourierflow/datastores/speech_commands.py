import os
import pathlib
import pickle
import tarfile
import urllib.request

import h5py
import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

from fourierflow.registry import Datastore


@Datastore.register('speech_commands')
class SpeechCommandsDatastore(Datastore):
    name = 'speech'

    def __init__(self, data_dir: str, train_id_path: str, test_id_path: str,
                 batch_size: int, n_workers: int):
        super().__init__()
        self.n_workers = n_workers
        self.batch_size = batch_size
        if not os.path.exists(data_dir):
            self._download(data_dir)
        self.X, self.y = self._preprocess(data_dir)
        # self.X.shape == [n_samples, seq_len, n_channels]
        # self.y.shape == [n_samples]

        views = h5py.File(data_dir, 'r')['views'][...]

        # Forward-fill missing values
        mask = views == -1
        idx = np.where(~mask, np.arange(mask.shape[1]), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        views = views[np.arange(idx.shape[0])[:, None], idx]

        # Fill remaining missing values with 0
        views[views == -1] = 0

        assert (views >= 0).all()

        with open(train_id_path, 'rb') as f:
            train_idx = sorted(pickle.load(f))
            train_views = views[train_idx]

        with open(test_id_path, 'rb') as f:
            test_idx = sorted(pickle.load(f))
            test_views = views[test_idx]

        self.train_dataset = VevoDataset(train_views[:, :49])
        self.valid_dataset = VevoDataset(test_views[:, 7:56])
        self.test_dataset = VevoDataset(test_views[:, 14:63])
        # train_dataset.shape == [1000, 64, 64, 10]

    def train_dataloader(self) -> DataLoader:
        loader = DataLoader(self.train_dataset,
                            batch_size=self.batch_size,
                            shuffle=True,
                            num_workers=self.n_workers,
                            drop_last=False,
                            pin_memory=True)
        return loader

    def val_dataloader(self) -> DataLoader:
        loader = DataLoader(self.valid_dataset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=self.n_workers,
                            drop_last=False,
                            pin_memory=True)
        return loader

    def test_dataloader(self) -> DataLoader:
        loader = DataLoader(self.test_dataset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=self.n_workers,
                            drop_last=False,
                            pin_memory=True)
        return loader

    def _download(self, data_dir):
        data_dir = pathlib.Path(data_dir)
        archive_path = data_dir / 'speech_commands.tar.gz'
        url = 'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
        urllib.request.urlretrieve(url, archive_path)
        with tarfile.open(archive_path, 'r') as f:
            f.extractall(data_dir)

    def _preprocess(self, data_dir):
        X = torch.empty(34975, 16000, 1)
        y = torch.empty(34975, dtype=torch.long)

        batch_index = 0
        y_index = 0
        for folder in ('yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'):
            loc = data_dir / folder
            for filename in os.listdir(loc):
                audio, _ = torchaudio.load_wav(loc / filename, channels_first=False,
                                               normalization=False)  # for forward compatbility if they fix it
                # Normalization argument doesn't seem to work so we do it manually.
                audio = audio / 2 ** 15

                # A few samples are shorter than the full length; for simplicity we discard them.
                if len(audio) != 16000:
                    continue

                X[batch_index] = audio
                y[batch_index] = y_index
                batch_index += 1
            y_index += 1
        assert batch_index == 34975, "batch_index is {}".format(batch_index)

        X = torchaudio.transforms.MFCC(log_mels=True, n_mfcc=20,
                                       melkwargs=dict(n_fft=200, n_mels=64))(X.squeeze(-1)).transpose(1, 2).detach()
        # X is of shape (batch=34975, length=161, channels=20)

        return X, y


class VevoDataset(Dataset):
    def __init__(self, views):
        self.views = torch.from_numpy(views).float()
        self.log_views = torch.log1p(self.views.clamp(min=0))

    def __len__(self):
        return self.views.shape[0]

    def __getitem__(self, idx):
        return (self.views[idx], self.log_views[idx])
