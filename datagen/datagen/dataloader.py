import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

class ProgramNpyDataset(Dataset):
    def __init__(self, npy_filepath, metadata_filepath, transform=None):
        self.npy_filepath = npy_filepath
        self.metadata_filepath = metadata_filepath
        self.transform = transform

        if not os.path.exists(npy_filepath):
            raise FileNotFoundError(f"Data file not found: {npy_filepath}")
        if not os.path.exists(metadata_filepath):
            raise FileNotFoundError(f"Metadata file not found: {metadata_filepath}")

        with open(metadata_filepath, 'r') as f:
            self.metadata = json.load(f)
        
        self.id_to_token = self.metadata["id_to_token"]
        self.token_to_id = {token: i for i, token in enumerate(self.id_to_token)}
        
        self.single_state_token_len = self.metadata["single_state_token_length"]
        self.record_token_len = self.metadata["record_token_length"]

        try:
            self.data = np.load(npy_filepath, mmap_mode='r')
        except ValueError as e:
            print(f"Warning: Could not load {npy_filepath}, possibly empty or corrupted: {e}. Treating as empty dataset.")
            self.data = np.array([], dtype=np.int32).reshape(0, self.record_token_len if self.record_token_len > 0 else 1)


        if self.data.ndim == 1 and self.data.shape[0] == 0 :
             self.data = self.data.reshape(0, self.record_token_len if self.record_token_len > 0 else 1)

        if self.data.ndim != 2:
             raise ValueError(f"Data in {npy_filepath} is not 2D. Shape: {self.data.shape}")
        if self.data.shape[0] > 0 and self.data.shape[1] != self.record_token_len:
            raise ValueError(f"Data record length {self.data.shape[1]} in {npy_filepath} does not match expected {self.record_token_len} from metadata.")


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if not (0 <= idx < len(self)):
             raise IndexError(f"Index {idx} out of bounds for dataset of length {len(self)}")

        record_tokens_np = self.data[idx].astype(np.int64)
        sample = torch.from_numpy(record_tokens_np)

        if self.transform:
            sample = self.transform(sample)
        return sample

if __name__ == '__main__':
    output_base_dir = "./20M_rows/"
    meta_files = [os.path.join(output_base_dir, f) 
                  for f in os.listdir(output_base_dir) 
                  if f.endswith(".meta.json")]
    
    if not meta_files:
        print(f"No .meta.json files found in {output_base_dir}. Please generate data first.")
        exit()

    list_of_datasets = []
    for meta_filepath in meta_files:
        npy_filepath = meta_filepath.replace(".meta.json", ".npy")
        if os.path.exists(npy_filepath):
            print(f"Loading dataset: {npy_filepath} with metadata {meta_filepath}")
            try:
                dataset_part = ProgramNpyDataset(npy_filepath, meta_filepath)
                if len(dataset_part) > 0:
                    list_of_datasets.append(dataset_part)
                else:
                    print(f"Skipping empty dataset: {npy_filepath}")
            except Exception as e:
                print(f"Error loading dataset from {npy_filepath}: {e}")
        else:
            print(f"Warning: Corresponding .npy file not found for {meta_filepath}")

    if not list_of_datasets:
        print("No data successfully loaded into datasets.")
        exit()

    combined_dataset = ConcatDataset(list_of_datasets)
    print(f"\nTotal records in combined dataset: {len(combined_dataset)}")

    if len(combined_dataset) > 0:
        first_record_tensor = combined_dataset[0]
        print(f"Shape of first record tensor: {first_record_tensor.shape}")
        print(f"First 20 tokens of first record: {first_record_tensor[:20].tolist()}")

        batch_size = 4
        dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

        print("\nIterating through DataLoader (first batch):")
        for i_batch, batch_data_cat_records in enumerate(dataloader):
            print(f"Batch {i_batch + 1}:")
            print(f"  Data shape: {batch_data_cat_records.shape}")
            break
