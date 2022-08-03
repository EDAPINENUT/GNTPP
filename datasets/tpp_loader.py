import torch
from models.libs.utils import one_hot_embedding
import numpy as np
from torch.utils.data import Dataset
import pickle
import torch.utils.data as data_utils
from pathlib import Path
import os
import torch.nn.functional as F

def load_dataset(dataset_dir, batch_size, val_batch_size=None, scale_normalization=50.0, device=None, **kwargs):
    print('loading datasets...')

    if val_batch_size == None:
        val_batch_size = batch_size
    
    train_set = SequenceDataset(
        dataset_dir, mode='train', batch_size=batch_size, scale_normalization=scale_normalization, device=device
    )

    validation_set = SequenceDataset(
        dataset_dir, mode='val', batch_size=val_batch_size, scale_normalization=scale_normalization, device=device
    )

    test_set = SequenceDataset(
        dataset_dir, mode='test', batch_size=val_batch_size, scale_normalization=scale_normalization, device=device
    )
    
    max_t_normalization = train_set.max_t
    for dataset in [train_set, validation_set, test_set]:
        setattr(dataset, 'max_t_normalization', max_t_normalization)

    mean_in_train, std_in_train = train_set.get_time_statistics()
    
    train_set.normalize(mean_in_train, std_in_train)
    validation_set.normalize(mean_in_train, std_in_train)
    test_set.normalize(mean_in_train, std_in_train)

    mean_log_dt, std_log_dt, max_dt = train_set.get_dt_statistics()
    
    data = {}
    data['train_loader'] = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate)
    data['val_loader']  = torch.utils.data.DataLoader(validation_set, batch_size=val_batch_size, shuffle=False, collate_fn=collate)
    data['test_loader'] = torch.utils.data.DataLoader(test_set, batch_size=val_batch_size, shuffle=False, collate_fn=collate)

    max_t = max([train_set.max_t, validation_set.max_t, test_set.max_t])/max_t_normalization*scale_normalization \
        if scale_normalization != 0 else max([train_set.max_t, validation_set.max_t, test_set.max_t])
    
    max_length = max([train_set.seq_lengths.max(), validation_set.seq_lengths.max(), test_set.seq_lengths.max()])

    assert train_set.event_type_num == validation_set.event_type_num == test_set.event_type_num
    event_type_num = train_set.event_type_num

    return data, event_type_num, {'train':train_set.seq_lengths, 'val':validation_set.seq_lengths, 'test': test_set.seq_lengths}, \
        max_length.item(), max_t, mean_log_dt.item(), std_log_dt.item(), max_dt.item()


class SequenceDataset(data_utils.Dataset):
    """Dataset class containing variable length sequences.

    Args:
        delta_times: Inter-arrival times between events. List of variable length sequences.

    """
    def __init__(self, dataset_dir, mode, batch_size, device=None, scale_normalization=50.0):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        else:
            self.device = device

        self.file = dataset_dir + mode + '_manifold_format.pkl'
        self.bs = batch_size
        self.scale_normalization = scale_normalization

        if os.path.exists(dataset_dir + '/granger_graph.npy'):
            self.granger_graph = np.load(dataset_dir + '/granger_graph.npy')

        with open(self.file, 'rb') as f:
            data = pickle.load(f)
        self.data = data
        self.process_data()



    def process_data(self):
        # print('processing dataset and saving in {}...'.format(self.processed_path))

        self.seq_times, self.seq_types, self.seq_lengths, self.seq_dts, self.max_t, self.event_type_num = \
            self.data['timestamps'], self.data['types'], self.data['lengths'], self.data['intervals'], self.data['t_max'], self.data['event_type_num']
        
        self.max_t = np.concatenate(self.data['timestamps']).max()
        self.seq_lengths = torch.Tensor(self.seq_lengths)

        self.in_times = [torch.Tensor(t[:-1]).float() for t in self.seq_times]
        self.out_times = [torch.Tensor(t[1:]).float() for t in self.seq_times]
        
        self.in_dts = [torch.Tensor(dt[:-1]).float() for dt in self.seq_dts]
        self.out_dts = [torch.Tensor(dt[1:]).float() for dt in self.seq_dts]

        self.in_types = [torch.Tensor(m[:-1]).long() for m in self.seq_types]
        self.out_types = [torch.Tensor(m[1:]).long() for m in self.seq_types]

        self.validate_times()



    @property
    def num_series(self):
        return len(self.in_times)

    def get_time_statistics(self):
        flat_in_times = torch.cat(self.in_times)
        return flat_in_times.mean(), flat_in_times.std()
    
    def get_dt_statistics(self):
        flat_in_dts_log = (torch.cat(self.in_dts)+ 1e-8).log()
        return flat_in_dts_log.mean(), flat_in_dts_log.std(), flat_in_dts_log.exp().max()
    
    def validate_times(self):
        if len(self.in_times) != len(self.out_times):
            raise ValueError("in_times and out_times have different lengths.")

        for s1, s2, s3, s4 in zip(self.in_times, self.out_times, self.in_types, self.out_types):
            if len(s1) != len(s2) or len(s3) != len(s4):
                raise ValueError("Some in/out series have different lengths.")
            if s3.max() >= self.event_type_num or s4.max() >= self.event_type_num:
                raise ValueError("Marks should not be larger than number of classes.")

    def normalize(self, mean_in=None, std_in=None, force_norm=False):
        """Apply mean-std normalization to times."""
        if mean_in is None or std_in is None:
            mean_in, std_in = self.get_mean_std_in()
            
        if force_norm:
            self.in_times = [(t - mean_in) / std_in for t in self.in_times]
            self.in_dts = [(t - mean_in) / std_in for t in self.in_dts]

        if self.scale_normalization != 0:
            self.in_times = [t / self.max_t_normalization * self.scale_normalization for t in self.in_times]
            self.in_dts = [t / self.max_t_normalization * self.scale_normalization for t in self.in_dts]

            self.out_times = [t / self.max_t_normalization * self.scale_normalization for t in self.out_times]
            self.out_dts= [t / self.max_t_normalization * self.scale_normalization for t in self.out_dts]

        return mean_in, std_in

    def get_mean_std_in(self):
        """Get mean and std of in_times."""
        flat_in_times = torch.cat(self.in_times)
        return flat_in_times.mean(), flat_in_times.std()

    def get_mean_std_out(self):
        """Get mean and std of out_times."""
        flat_out_times = torch.cat(self.out_times)
        return flat_out_times.mean(), flat_out_times.std()

    def get_log_mean_std_out(self):
        """Get mean and std of out_times."""
        flat_out_times = torch.cat(self.out_times).log()
        return flat_out_times.mean(), flat_out_times.std()

    def __getitem__(self, key):
        return self.in_dts[key], self.out_dts[key], self.in_types[key], self.out_types[key],\
            self.in_times[key], self.seq_lengths[key], self.event_type_num, self.device

    def __len__(self):
        return self.num_series

    def __repr__(self):
        return f"SequenceDataset({self.num_series})"

def collate(batch):

    device = batch[0][7]
    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    in_dts = [item[0] for item in batch]
    out_dts = [item[1] for item in batch]
    in_types = [item[2] for item in batch]
    out_types = [item[3] for item in batch]
    in_times = [item[4] for item in batch]
    
    seq_lengths = torch.Tensor([item[5] for item in batch])
    event_type_num = batch[0][6]

    in_dts = torch.nn.utils.rnn.pad_sequence(in_dts, batch_first=True, padding_value=0.0)
    out_dts = torch.nn.utils.rnn.pad_sequence(out_dts, batch_first=True, padding_value=0.0)
    in_types = torch.nn.utils.rnn.pad_sequence(in_types, batch_first=True, padding_value=event_type_num)
    out_types = torch.nn.utils.rnn.pad_sequence(out_types, batch_first=True, padding_value=event_type_num)
    in_times = torch.nn.utils.rnn.pad_sequence(in_times, batch_first=True, padding_value=0.0)
    
    matrix_batch = in_times[:,:,None] - in_times[:,None,:]
    matrix_batch[matrix_batch<0] = 0.0

    out_onehots = one_hot_embedding(out_types, event_type_num + 1)[:,:,:-1]
    
    return Batch(
        in_dts.to(device), 
        in_types.to(device),
        in_times.to(device),
        matrix_batch.to(device),
        seq_lengths.to(device), 
        out_dts.to(device), 
        out_types.to(device),
        out_onehots.to(device)
        )


class Batch():
    def __init__(self, in_dts, in_types, in_times, matrix_batch, seq_lengths, out_dts, out_types, out_onehots):
        self.in_dts = in_dts
        self.in_types = in_types.long()
        self.in_times = in_times
        self.lag_matrixes = matrix_batch
        self.seq_lengths = seq_lengths
        self.out_dts = out_dts
        self.out_types = out_types.long()
        self.out_onehots = out_onehots.long()