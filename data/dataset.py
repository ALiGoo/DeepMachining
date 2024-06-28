import math
import numpy as np
import torch
from torch.utils.data import Dataset

def sampling_signal(data, training):
    signals_list = list()
    sampling_index = list()
    for piece_idx, filepath in enumerate(data.filepaths.values):
        signals = np.load(filepath)
        
        signal_length = signals.shape[2]
        index_5mm = math.floor(signal_length / 215 * 5) - 1
        index_75mm = math.floor(signal_length / 215 * 75) - 1
        index_145mm = math.floor(signal_length / 215 * 145) - 1
        if training:
            label_index = np.arange(index_5mm, index_145mm, 10240)
        else:    
            label_index = np.array([index_5mm, index_75mm, index_145mm])
        
        sampling_index.append(np.stack([np.full(len(label_index), piece_idx), label_index]))
        signals_list.append(signals)

    sampling_index = np.concatenate(sampling_index, axis=1).transpose((1,0))
    return signals_list, sampling_index


class Doe2Dataset(Dataset):
    def __init__(self, data, training=True):
        self.signals, self.sampling_index = sampling_signal(data, training)
    
    def __len__(self):
        return len(self.sampling_index)
    
    def __getitem__(self, idx):
        piece_idx = self.sampling_index[idx][0]
        sequence_idx = self.sampling_index[idx][1]
        signal = self.signals[piece_idx][:,1:3,sequence_idx-5120:sequence_idx+5120].copy()
        # delete sound channel
        # signal = np.delete(signal, 3, axis=1)
        signal = np.transpose(signal, (1, 0, 2)).reshape((signal.shape[1],-1))
        label = self.signals[piece_idx][1,-1,sequence_idx]
        
        signal = torch.tensor(signal, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        label = label - 44.6

        return signal, label.unsqueeze(0)