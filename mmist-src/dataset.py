import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from torch.utils.data.dataloader import default_collate

def collate_fn(batch):
    out = {}
    keys = batch[0].keys()
    max_wsi_len = max([item['input_pathology'].shape[0] for item in batch]) if 'input_pathology' in keys else 0
    max_ct_len = max([item['input_CT'].shape[0] for item in batch]) if 'input_CT' in keys else 0
    
    for key in keys:
        if key == 'input_pathology':
            padded = []
            for item in batch:
                pad_len = max_wsi_len - item[key].shape[0]
                if pad_len > 0:
                    padded.append(F.pad(item[key], (0, 0, 0, pad_len), "constant", 0))
                else:
                    padded.append(item[key])
            out[key] = torch.stack(padded, dim=0)
        elif key == 'input_CT':
            padded = []
            for item in batch:
                pad_len = max_ct_len - item[key].shape[0]
                if pad_len > 0:
                    padded.append(F.pad(item[key], (0, 0, 0, pad_len), "constant", 0))
                else:
                    padded.append(item[key])
            out[key] = torch.stack(padded, dim=0)
        else:
            try:
                out[key] = default_collate([item[key] for item in batch])
            except Exception:
                out[key] = [item[key] for item in batch]
    return out

class ImageDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.args = args
        self.mode = mode
        
        # Base feature directory
        # e.g., args.path_feature_CT points to /mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/LLM-guided-Multimodal-MIL/feat/feat-MMIST
        # We assume the base dir is the parent of master_dataset.csv
        self.base_dir = '/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/LLM-guided-Multimodal-MIL/feat/feat-MMIST'
        
        # Load master dataframe
        self.df = pd.read_csv(os.path.join(self.base_dir, 'master_dataset.csv'))
        
        # Filter based on mode (train/val combined vs test)
        if self.mode == 'test':
            self.df = self.df[self.df['Split'] == 'test'].reset_index(drop=True)
        else:
            # Ablation run trains on train+val
            self.df = self.df[self.df['Split'].isin(['train', 'val'])].reset_index(drop=True)
            
        self.df['case_id'] = self.df['case_id'].astype(str)
        self.df['label'] = self.df['vital_status_12'].fillna(0).astype(int) # 1 alive, 0 death

        # Build CT dictionary mapping: Case ID -> Series Instance UID 
        tcga_df = pd.read_csv(os.path.join(self.base_dir, 'TCGA.csv'))
        cptac_df = pd.read_csv(os.path.join(self.base_dir, 'CPTAC.csv'))
        self.ct_mapping_df = pd.concat([tcga_df, cptac_df], ignore_index=True)
        self.ct_id_to_uid = {}
        for _, row in self.ct_mapping_df.iterrows():
            uid = str(row['Series Instance UID'])
            pid = str(row['Patient ID'])
            self.ct_id_to_uid[pid] = uid

        # Initialize available files mapped by case_id
        # We index the available specific feature files per modality to avoid FileNotFoundError
        self.ct_files = self._index_ct_files()
        self.wsi_files = self._index_wsi_files()
        self.clinical_files = self._index_clinical_files()

        # Remove cases where ALL requested modalities are strictly missing from the dataset storage
        # (Though we zero them out if partially missing)
        valid_indices = []
        for i in range(len(self.df)):
            case_id = self.df.iloc[i]['case_id']
            has_ct = case_id in self.ct_files
            has_wsi = case_id in self.wsi_files
            has_clinical = case_id in self.clinical_files
            if has_ct or has_wsi or has_clinical:
                valid_indices.append(i)
        
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)

    def _index_ct_files(self):
        feat_dir = os.path.join(self.base_dir, 'CT', 'train' if self.mode != 'test' else 'test')
        # Some test instances might overlap into train folder if we merged train+val, so let's glob both cleanly:
        all_files = glob_all_splits(os.path.join(self.base_dir, 'CT'))
        
        ct_map = {}
        # We map uid -> patient_id
        uid_to_pid = {v: k for k, v in self.ct_id_to_uid.items()}
        for f in all_files:
            fname = os.path.basename(f).replace('.pt', '')
            uid = fname.split('_')[0] # ignore _* suffix
            if uid in uid_to_pid:
                ct_map[uid_to_pid[uid]] = f
        return ct_map

    def _index_wsi_files(self):
        all_files = glob_all_splits(os.path.join(self.base_dir, 'WSI'))
        wsi_map = {}
        for f in all_files:
            fname = os.path.basename(f).replace('.pt', '')
            if '-' in fname:
                case_id = '-'.join(fname.split('-')[:-1])
            else:
                case_id = fname
            wsi_map[case_id] = f
        return wsi_map

    def _index_clinical_files(self):
        all_files = glob_all_splits(os.path.join(self.base_dir, 'Clinical'))
        clin_map = {}
        for f in all_files:
            case_id = os.path.basename(f).replace('.pt', '')
            clin_map[case_id] = f
        return clin_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        case_id = row['case_id']
        label = row['label']
        label_onehot = torch.nn.functional.one_hot(torch.tensor(label).to(torch.int64), num_classes=self.args.num_classes)

        # Modalities combinations (ablation_mode): e.g. "CT_WSI_Clinical", "CT_WSI", "CT_Clinical"
        # Extract individual flags from the ablation_mode string
        req_ct = 'CT' in self.args.ablation_mode
        req_wsi = 'WSI' in self.args.ablation_mode
        req_clin = 'Clinical' in self.args.ablation_mode
        
        # Load or generate zero-tensors (1, 768)
        dummy_shape = (1, 768)
        
        if req_ct and case_id in self.ct_files:
            loaded_ct = torch.load(self.ct_files[case_id])
            if isinstance(loaded_ct, dict):
                input_CT = loaded_ct['features'].float()
            else:
                input_CT = loaded_ct.float()
        else:
            input_CT = torch.zeros(dummy_shape, dtype=torch.float32)

        if req_wsi and case_id in self.wsi_files:
            loaded_wsi = torch.load(self.wsi_files[case_id])
            if isinstance(loaded_wsi, dict):
                input_wsi = loaded_wsi['features'].float()
            else:
                input_wsi = loaded_wsi.float()
        else:
            input_wsi = torch.zeros(dummy_shape, dtype=torch.float32)

        if req_clin and case_id in self.clinical_files:
            try:
                loaded_clin = torch.load(self.clinical_files[case_id])
                if isinstance(loaded_clin, dict):
                    input_clinical = loaded_clin['features'].float()
                else:
                    input_clinical = loaded_clin.float()
                if input_clinical.dim() == 1:
                    input_clinical = input_clinical.unsqueeze(0)
            except Exception: # In case of corrupted feature
                input_clinical = torch.zeros(dummy_shape, dtype=torch.float32)
        else:
            input_clinical = torch.zeros(dummy_shape, dtype=torch.float32)

        # In current MMIST pipeline, `test_ddp.py` unpacks `input_CT`, `input_pathology`, `input_CI`
        # Because we'll adapt test_ddp to receive all 3 universally.
        data_dict = {
            'input_CT': input_CT,
            'input_pathology': input_wsi,
            'input_CI': input_clinical,
            'label': label_onehot,
            'ID': case_id
        }

        # Handle mask parameter for legacy architectures just in case
        if hasattr(self.args, 'model_CT') and 'wMask' in self.args.model_CT:
            data_dict['mask'] = torch.zeros((1, 224, 224)) # Dummy mask

        return data_dict

def glob_all_splits(directory):
    import glob
    files = []
    if os.path.exists(os.path.join(directory, 'train')):
        files.extend(glob.glob(os.path.join(directory, 'train', '*.pt')))
    if os.path.exists(os.path.join(directory, 'val')):
        files.extend(glob.glob(os.path.join(directory, 'val', '*.pt')))
    if os.path.exists(os.path.join(directory, 'test')):
        files.extend(glob.glob(os.path.join(directory, 'test', '*.pt')))
    # Also grab from root just in case
    files.extend(glob.glob(os.path.join(directory, '*.pt')))
    return files