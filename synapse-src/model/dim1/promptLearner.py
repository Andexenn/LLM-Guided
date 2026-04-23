import os.path as osp
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

# clip is disabled
# import clip

class PromptLearner(nn.Module):
    def __init__(self, args, text):
        super(PromptLearner, self).__init__()
        self.args = args
        # CLIP is disabled as it is no longer used and library files are missing.
        
    def forward(self):
        # Return dummy prompts or handle as needed
        return torch.zeros((1, 1, 512))