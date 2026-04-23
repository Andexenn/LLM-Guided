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
        
        self.base_dir = '/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/LLM-guided-Multimodal-MIL/feat/feat-SYNAPSE'
        self.df = pd.read_csv(os.path.join(self.base_dir, 'cleaned_split_dataset.csv'))
        
        # Filter based on mode
        if self.mode == 'test':
            self.df = self.df[self.df['split'] == 'test'].reset_index(drop=True)
        else:
            self.df = self.df[self.df['split'].isin(['train', 'val'])].reset_index(drop=True)
            
        self.df['label'] = self.df['label'].fillna(0).astype(int) # 1 alive, 0 death
        # Fill NA mappings with empty strings
        for col in ['dmp_pt_id', 'radiology_accession_number', 'slide_id']:
            self.df[col] = self.df[col].fillna('')

        # Initialize available files
        self.ct_files = self._index_ct_files()
        self.wsi_files = self._index_wsi_files()
        self.clinical_files = self._index_clinical_files()

        # Remove cases where ALL specific mapped modalities are entirely unfound
        valid_indices = []
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            ct_id = str(row['radiology_accession_number'])
            wsi_id = str(row['slide_id']).split('.')[0] if '.' in str(row['slide_id']) else str(row['slide_id'])
            clin_id = str(row['dmp_pt_id'])
            
            has_ct = ct_id in self.ct_files and ct_id != ''
            has_wsi = wsi_id in self.wsi_files and wsi_id != ''
            has_clinical = clin_id in self.clinical_files and clin_id != ''
            if has_ct or has_wsi or has_clinical:
                valid_indices.append(i)
        
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)

    def _index_ct_files(self):
        all_files = glob_all_splits(os.path.join(self.base_dir, 'CT'))
        ct_map = {}
        for f in all_files:
            fname = os.path.basename(f).replace('.pt', '')
            ct_map[fname] = f
        return ct_map

    def _index_wsi_files(self):
        all_files = glob_all_splits(os.path.join(self.base_dir, 'WSI'))
        wsi_map = {}
        for f in all_files:
            fname = os.path.basename(f).replace('.pt', '')
            wsi_map[fname] = f
        return wsi_map

    def _index_clinical_files(self):
        all_files = glob_all_splits(os.path.join(self.base_dir, 'Clinical'))
        clin_map = {}
        for f in all_files:
            fname = os.path.basename(f).replace('.pt', '')
            clin_map[fname] = f
        return clin_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        ct_id = str(row['radiology_accession_number'])
        wsi_id = str(row['slide_id']).split('.')[0] if '.' in str(row['slide_id']) else str(row['slide_id'])
        clin_id = str(row['dmp_pt_id'])
        
        label = row['label']
        label_onehot = torch.nn.functional.one_hot(torch.tensor(label).to(torch.int64), num_classes=self.args.num_classes)

        # Modalities combinations (ablation_mode)
        req_ct = 'CT' in self.args.ablation_mode
        req_wsi = 'WSI' in self.args.ablation_mode
        req_clin = 'Clinical' in self.args.ablation_mode
        
        dummy_shape = (1, 768)
        
        if req_ct and ct_id in self.ct_files and ct_id != '':
            try:
                loaded_ct = torch.load(self.ct_files[ct_id])
                if isinstance(loaded_ct, dict):
                    input_CT = loaded_ct['features'].float()
                else:
                    input_CT = loaded_ct.float()
            except Exception:
                input_CT = torch.zeros(dummy_shape, dtype=torch.float32)
        else:
            input_CT = torch.zeros(dummy_shape, dtype=torch.float32)

        if req_wsi and wsi_id in self.wsi_files and wsi_id != '':
            try:
                loaded_wsi = torch.load(self.wsi_files[wsi_id])
                if isinstance(loaded_wsi, dict):
                    input_wsi = loaded_wsi['features'].float()
                else:
                    input_wsi = loaded_wsi.float()
            except Exception:
                input_wsi = torch.zeros(dummy_shape, dtype=torch.float32)
        else:
            input_wsi = torch.zeros(dummy_shape, dtype=torch.float32)

        if req_clin and clin_id in self.clinical_files and clin_id != '':
            try:
                loaded_clin = torch.load(self.clinical_files[clin_id])
                if isinstance(loaded_clin, dict):
                    input_clinical = loaded_clin['features'].float()
                else:
                    input_clinical = loaded_clin.float()
                if input_clinical.dim() == 1:
                    input_clinical = input_clinical.unsqueeze(0)
            except Exception:
                input_clinical = torch.zeros(dummy_shape, dtype=torch.float32)
        else:
            input_clinical = torch.zeros(dummy_shape, dtype=torch.float32)

        data_dict = {
            'input_CT': input_CT,
            'input_pathology': input_wsi,
            'input_CI': input_clinical,
            'label': label_onehot,
            'ID': clin_id or wsi_id or ct_id
        }

        # Handling legacy mask
        if hasattr(self.args, 'model_CT') and 'wMask' in self.args.model_CT:
            data_dict['mask'] = torch.zeros((1, 224, 224))

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
    # Root
    files.extend(glob.glob(os.path.join(directory, '*.pt')))
    return files