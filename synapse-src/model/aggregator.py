import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math

from .sam.transformer import TwoWayTransformer

class aggregator(nn.Module):
    def __init__(self, args):
        super(aggregator, self).__init__()
        
        self.args = args
        
        embedding_dim = 512
        
        if 'CT' in self.args.modality:
            weights = 'DEFAULT'
            progress = True
            self.extractor_CT = nn.Sequential(nn.Linear(768, embedding_dim), nn.Tanh())
            
            self.TwoWayTransformer_CT = TwoWayTransformer(
                args=self.args,
                depth=2,
                embedding_dim=embedding_dim,
                num_heads=8,
                mlp_dim=2048,
            )
            
        self.fc_CI2CT = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.Tanh())
                
        if 'pathology' in self.args.modality:
            self.fc_pathology = nn.Sequential(nn.Linear(768, embedding_dim), nn.Tanh())
            # Pre-extracted features (768-D) skip raw extractor.
            self.extractor_pathology = nn.Sequential(nn.Linear(768, embedding_dim), nn.Tanh())
            
            self.TwoWayTransformer_Pth = TwoWayTransformer(
                args=self.args,
                depth=2,
                embedding_dim=embedding_dim,
                num_heads=8,
                mlp_dim=2048,
            )
            
        self.fc_CI2Pth = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.Tanh())

        self.fc_CI = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.Tanh())
        
        self.TwoWayTransformer_Both = TwoWayTransformer(
            args=self.args,
            depth=2,
            embedding_dim=embedding_dim,
            num_heads=8,
            mlp_dim=2048,
        )
        
        
        if self.args.aggregator == 'ABMIL':
            from .dim1 import ABMIL
            self.aggregator = ABMIL(args, L=embedding_dim)
        elif self.args.aggregator == 'ABMIL_v2':
            from .dim1 import ABMIL_v2
            self.aggregator = ABMIL_v2(args, L=embedding_dim)
        elif self.args.aggregator == 'TransMIL':
            from .dim1 import TransMIL
            self.aggregator = TransMIL(n_classes=self.args.num_classes, L=embedding_dim)
        elif self.args.aggregator == 'TransMIL_seperate':
            from .dim1 import TransMIL
            if 'CT' in self.args.modality:
                self.aggregator_CT = TransMIL(n_classes=self.args.num_classes, L=embedding_dim)
            if 'pathology' in self.args.modality:
                self.aggregator_Pth = TransMIL(n_classes=self.args.num_classes, L=embedding_dim)
            # self.aggregator = TransMIL(n_classes=self.args.num_classes, L=embedding_dim)
            from .dim1 import ABMIL
            self.aggregator = ABMIL(args, L=embedding_dim)
        
        
        # max_seq_len = 15592
        max_seq_len = 100000
        self.pe = torch.zeros((max_seq_len, embedding_dim))
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, embedding_dim, 2, dtype=torch.float) * -(math.log(10000.0) / embedding_dim)))
        self.pe[:, 0::2] = torch.sin(position.float() * div_term)
        self.pe[:, 1::2] = torch.cos(position.float() * div_term)
        self.pe = self.pe.unsqueeze(0)
        
        self.clinic_extractor = nn.Sequential(nn.Linear(768, embedding_dim), nn.Tanh())
        # self.prompt = nn.Parameter(torch.randn(2, self.args.prompt_len, embedding_dim))
        self.prompt_embedding = nn.Parameter(torch.randn(1, embedding_dim))
        # self.prompt_embedding = nn.Embedding(num_embeddings=4, embedding_dim=embedding_dim)
        
        
        self.fc_concat = nn.Linear(len(self.args.modality) * embedding_dim, embedding_dim)
        
        self.fc = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(embedding_dim, self.args.num_classes)
        )
            
    
    def _truncate_seq(self, x, max_len):
        """Truncate sequence to max_len by uniform subsampling if too long."""
        B, N, D = x.shape
        if N <= max_len:
            return x
        indices = torch.linspace(0, N - 1, max_len).long().to(x.device)
        return x[:, indices, :]
        
    def forward(self, x_list, x_CI):
        # x_list = [x_CT, x_pathology] CT/pathology 순서대로
        # x_CI : (1, 77-prompt_len)
        
        max_len = getattr(self.args, 'max_seq_len_agg', 1024)
        
        # --- Extract and truncate features with consistent variable names ---
        x_input_CT = None
        x_input_pathology = None
        
        if ('CT' in self.args.modality) and ('pathology' in self.args.modality):
            x_input_CT = self.extractor_CT(x_list[0])
            x_input_CT = self._truncate_seq(x_input_CT, max_len)
            x_input_pathology = self.fc_pathology(x_list[1])
            x_input_pathology = self._truncate_seq(x_input_pathology, max_len)
        elif 'CT' in self.args.modality:
            x_input_CT = self.extractor_CT(x_list[0])
            x_input_CT = self._truncate_seq(x_input_CT, max_len)
        elif 'pathology' in self.args.modality:
            x_input_pathology = self.fc_pathology(x_list[0])
            x_input_pathology = self._truncate_seq(x_input_pathology, max_len)
        
        x_CI_prompted = self.clinic_extractor(x_CI) # (1, 1, 512)
        
        # x_CI_prompted = torch.cat([self.prompt_embedding.unsqueeze(0).expand(x_CI_prompted.shape[0], -1, -1), x_CI_prompted], dim=1)
        
        x_CT2CI = None
        x_Pth2CI = None
        
        if self.args.integration_method == 'concat':
            x0_list = []
            if 'CT' in self.args.modality: x0_list.append(x_input_CT.mean(dim=1))
            if 'pathology' in self.args.modality: x0_list.append(x_input_pathology.mean(dim=1))
            if 'CI' in self.args.modality: x0_list.append(x_CI_prompted.mean(dim=1))
            
            x0 = torch.cat(x0_list, dim=1)
            x0 = self.fc_concat(x0)
            
            x_CT2CI = x0.unsqueeze(1)
            x_Pth2CI = x0.unsqueeze(1)
            
        elif self.args.integration_method == 'G':
            x0_list = []
            if 'CT' in self.args.modality: x0_list.append(x_input_CT)
            if 'pathology' in self.args.modality: x0_list.append(x_input_pathology)
            if 'CI' in self.args.modality: x0_list.append(x_CI_prompted)
            
            x0 = torch.cat(x0_list, dim=1)
            
            x_CT2CI = x_input_CT if 'CT' in self.args.modality else x0
            x_Pth2CI = x_input_pathology if 'pathology' in self.args.modality else x0
            
            if self.args.aggregator != '-':
                x0 = self.aggregator(x0)
                if isinstance(x0, tuple): x0 = x0[0]
                
        else: # MAM_G
            if ('CT' in self.args.modality) and ('pathology' in self.args.modality):
                x_CT2CI, x_CI2CT = self.TwoWayTransformer_Both(x_input_CT, self.pe[:,:x_input_CT.shape[1]].cuda(), self.fc_CI2CT(x_CI_prompted))
                
                x_Pth2CI, x_CI2Pth = self.TwoWayTransformer_Both(x_input_pathology, self.pe[:,:x_input_pathology.shape[1]].cuda(), self.fc_CI2Pth(x_CI_prompted))
                
                if self.args.aggregator == 'TransMIL_seperate':
                    x_CI2CT = self.aggregator_CT(x_CI2CT)[:,None,:]
                    x_CI2Pth = self.aggregator_Pth(x_CI2Pth)[:,None,:]
                x0 = torch.cat([x_CT2CI, x_CI2CT, x_Pth2CI, x_CI2Pth], dim=1)
            
            elif ('CT' in self.args.modality):
                x_CT2CI, x_CI2CT = self.TwoWayTransformer_CT(x_input_CT, self.pe[:,:x_input_CT.shape[1]].cuda(), self.fc_CI2CT(x_CI_prompted))
                x0 = torch.cat([x_CT2CI, x_CI2CT], dim=1)
    
            elif ('pathology' in self.args.modality):
                x_Pth2CI, x_CI2Pth = self.TwoWayTransformer_Pth(x_input_pathology, self.pe[:,:x_input_pathology.shape[1]].cuda(), self.fc_CI2Pth(x_CI_prompted))
                x0 = torch.cat([x_Pth2CI, x_CI2Pth], dim=1)
            
            elif ('CI' in self.args.modality):
                x0 = self.fc_CI(x_CI_prompted)
            
            if self.args.aggregator != '-':
                x0 = self.aggregator(x0)
                if isinstance(x0, tuple):
                    x0 = x0[0]
        x = torch.sigmoid(self.fc(x0))
        
        if ('CT' in self.args.modality) and ('pathology' in self.args.modality):
            return x, x_CT2CI, x_Pth2CI
        elif ('CT' in self.args.modality):
            return x, x_CT2CI
        elif ('pathology' in self.args.modality):
            return x, x_Pth2CI
        elif ('CI' in self.args.modality):
            return x