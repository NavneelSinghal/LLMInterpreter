import os
import json
import numpy as np
import torch

from torch.utils.data import Dataset

class StateTransitionDataset(Dataset):
    def __init__(self, data_dir):
        self.mask = None
        self.single_state_token_length = None
        self.record_token_length = None
        self.vocab_size = None
        self.id_to_token = None
        self.token_to_id = None
        self.program_params = None

        self.npy_and_meta_files = sorted([
            (os.path.join(data_dir, f), os.path.join(data_dir, f.replace('.npy', '.meta.json')))
            for f in os.listdir(data_dir) if f.endswith('.npy')
        ])
        
        if not self.npy_and_meta_files:
            raise FileNotFoundError(f"No .npy files found in {data_dir}")

        self.mmaps = []
        self.record_counts = []
        self.cumulative_record_counts = []
        
        first_meta_processed = False
        for npy_file, meta_file in self.npy_and_meta_files:
            try:
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)
                
                if not first_meta_processed:
                    self.record_token_length = metadata['record_token_length']
                    self.single_state_token_length = metadata['single_state_token_length']
                    self.id_to_token = metadata['id_to_token']
                    self.program_params = metadata['program_params']
                    self.vocab_size = len(self.id_to_token)
                    first_meta_processed = True
                else:
                    assert (self.record_token_length == metadata['record_token_length']), \
                        f"Metadata mismatch: record_token_length in {meta_file}"
                    assert (self.single_state_token_length == metadata['single_state_token_length']), \
                        f"Metadata mismatch: single_state_token_length in {meta_file}"
                    assert (self.id_to_token == metadata['id_to_token']), \
                        f"Metadata mismatch: id_to_token in {meta_file}"
                    assert (self.program_params == metadata['program_params']), \
                        f"Metadata mismatch: program_params in {meta_file}"
                
                mmap_arr = np.load(npy_file, mmap_mode='r')
                self.mmaps.append(mmap_arr)
                
                if mmap_arr.ndim != 2:
                    self.close()
                    raise RuntimeError(
                        f'Number of ndimensions for {npy_file} != 2, is {mmap_arr.ndim} instead'
                    )
                
                if mmap_arr.shape[1] != self.record_token_length:
                    self.close()
                    raise RuntimeError(
                        f'Metadata is faulty for {npy_file}, check record_token_length and dimensions of the data'
                    )
                
                self.record_counts.append(mmap_arr.shape[0])
                
            except Exception as e:
                self.close()
                print(f'Unable to read dataset file {npy_file} or {meta_file} due to Exception: {e}')
                raise
        
        if not self.mmaps:
            raise RuntimeError("No data files were successfully loaded or all were problematic.")

        self.cumulative_record_counts = np.cumsum(self.record_counts)
        self.n_records = self.cumulative_record_counts[-1] if len(self.cumulative_record_counts) > 0 else 0
        self.token_to_id = {token: i for i, token in enumerate(self.id_to_token)}
        
    def __len__(self):
        return self.n_records

    def __getitem__(self, idx):
        if not 0 <= idx < self.n_records:
            raise IndexError(f'Index {idx} is out of bounds for a dataset with length {self.n_records}')
        
        file_idx = np.searchsorted(self.cumulative_record_counts, idx, side='right')
        if file_idx > 0:
            idx -= self.cumulative_record_counts[file_idx - 1]
        
        return torch.from_numpy(np.array(self.mmaps[file_idx][idx])).long()

    def close(self):
        if hasattr(self, 'mmaps') and self.mmaps:
            print("\nClosing memory-mapped files...")
            for i, mmap_obj in enumerate(self.mmaps):
                file_basename = (os.path.basename(self.npy_and_meta_files[i][0]) 
                               if hasattr(self, 'npy_and_meta_files') and i < len(self.npy_and_meta_files) 
                               else f"file {i}")
                try:
                    if hasattr(mmap_obj, '_mmap') and mmap_obj._mmap is not None:
                        mmap_obj._mmap.close()
                        print(f"  Closed mmap for {file_basename} (via _mmap).")
                    elif (hasattr(mmap_obj, 'base') and isinstance(mmap_obj.base, np.memmap) 
                          and hasattr(mmap_obj.base, '_mmap')):
                        mmap_obj.base._mmap.close()
                        print(f"  Closed mmap for {file_basename} (via base._mmap).")
                except Exception as e:
                    print(f"  Error closing mmap for {file_basename}: {e}")
            self.mmaps = []
        else:
            print("No memory-mapped files to close or dataset not fully initialized.")

    def get_tokenizer(self):
        def tokenize(text):
            text_list = text.strip().split()
            return torch.tensor([self.token_to_id[x] for x in text_list], dtype=torch.long)
        
        def to_string(array_like):
            if isinstance(array_like, torch.Tensor):
                array_like = array_like.tolist()
            
            if not isinstance(array_like, list) or (array_like and not isinstance(array_like[0], int)):
                if hasattr(array_like, 'ndim') and array_like.ndim > 1:
                    raise RuntimeError('Tokenizer to_string input can have only 1 dimension of tokens')
            
            return ' '.join([self.id_to_token[x] for x in array_like])
        
        return tokenize, to_string
