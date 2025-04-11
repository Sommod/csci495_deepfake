'''
    This file contains the methods and information for creating a Visual Transformer (ViT) for the model
    that is used for video detection (GenConViT). Specically, this creates a Swin Transformer for the model.
    The reason for Swin vs a regular transformer is it provides the ability to connect windows, allowing
    communication between said windows.
'''

# Imports
import torch as T
import torch.nn as nn
import numpy as np
import einops as en
import scipy.special as SS

class SwinTEmbedding(nn.Module):
    '''
        Patch Embedding
    '''
    def __init__(self, patch_size = 4, emb_size = 96):
        super().__init__()
        self.linear_embed = nn.Conv2d(3, emb_size, kernel_size = patch_size, stride = patch_size)
        self.rearrange = en.rearrange(' b c h w => b (h w) c')

    def forward(self, value):
        value = self.linear_embed(value)
        value = self.rearrange(value)
        return value
    
class PatchMerging(nn.Module):
    '''
        Patch Merging
    '''
    def __init__(self, emb_size):
        super().__init__()
        self.linear = nn.Linear(4 * emb_size, 2 * emb_size)

    def forward(self, value):
        B, L, C = value.shape
        H = W = int(np.sqrt(L) / 2)
        value = en.rearrange(value, 'b (h sl w s2) c -> b (h w) (s1 s2 c)', s1 = 2, s2 = 2, h = H, w = W)
        value = self.linear(value)
        return value
    
class ShiftedWindow(nn.Module):
    '''
        Similar to the standard self-attention, but with the additional aspects of having shifted
        windows and masking.
    '''
    def __init__(self, emb_size, num_head, window_size = 7, shifted = True):
        super().__init__()
        self.emb_size = emb_size
        self.num_head = num_head
        self.window_size = window_size
        self.shifted = shifted
        self.linear_1 = nn.Linear(emb_size, 3 * emb_size)
        self.linear_2 = nn.Linear(emb_size, emb_size)

        self.pos_embeddings = nn.Parameter(T.randn(window_size * 2 - 1, window_size * 2 - 1))
        self.indices = T.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
        self.relative_indices = self.indices[None, :, :] - self.indices[:, None, :]
        self.relative_indices += self.window_size - 1

    def forward(self, value):
        head_dim = self.emb_size / self.num_head
        height = width = int(np.sqrt(value.shape[1]))

        value = self.linear_1(value)
        value = en.rearrange(value, 'b (h w) (c k) => b h w c k', h = height, w = width, k = 3, c = self.emb_size)

        if(self.shifted):
            value =  T.roll(value, (-self.window_size // 2, - self.window_size // 2), dims = (1, 2))
            value = en.rearrange(value, 'b (Wh w1) (Ww w2) (e H) k -> b H Wh Ww (w1 w2) e k', w1 = self.window_size, w2 = self.window_size, H = self.num_head)       

        Q, K, V = value.chunk(3, dim = 6)
        Q, K, V = Q.squeeze(-1), K.squeeze(-1), V.squeeze(-1)
        wei = (Q @ K.transpose(4,5)) / np.sqrt(head_dim)

        if(self.shifted):
            row_mask = T.zeros((self.window_size * 2, self.window_size * 2)).cuda()
            row_mask[-self.window_size * (self.window_size // 2):, 0:-self.window_size * (self.window_size // 2)] = float('-inf')
            row_mask[0:-self.window_size * (self.window_size // 2), -self.window_size * (self.window_size // 2):] = float('-inf')
            column_mask = en.rearrange(row_mask, '(r w1) (c w2) -> (w1 r) (w2 c)', w1 = self.window_size, w2 = self.window_size)
            wei[:, :, -1, :] += row_mask
            wei[:, :, :, -1] += column_mask
        
        wei = SS.softmax(wei, dim=-1) @ V
        value = en.rearrange(wei, 'b H Wh Ww (w1 w2) e -> b (Wh w1) (Ww w2) (H e)', w1 = self.window_size, w2 = self.window_size, H = self.num_heads)
        value = en.rearrange(value, 'b h w c -> b (h w) c')
        
        return self.linear_2(value)
    
class MLP(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.ff = nn.Sequential(nn.Linear(emb_size, 4 * emb_size), nn.GELU(), nn.Linear(4 * emb_size, emb_size),)
    
    def forward(self, value):
        return self.ff(value)
    
class SwinEncoder(nn.Module):
    '''
        Encoder for the Transformer
    '''
    def __init__(self, emb_size, num_heads, window_size = 7):
        super().__init__()
        self.WMSA = ShiftedWindow(emb_size, num_heads, window_size, shifted=False)
        self.SWMSA = ShiftedWindow(emb_size, num_heads, window_size, shifted=True)
        self.ln = nn.LayerNorm(emb_size)
        self.MLP = MLP(emb_size)
        
    def forward(self, x):
        # Window Attention
        x = x + self.WMSA(self.ln(x))
        x = x + self.MLP(self.ln(x))

        # shifted Window Attention
        x = x + self.SWMSA(self.ln(x))
        x = x + self.MLP(self.ln(x))
        
        return x
    
class Swin(nn.Module):
    '''
        Main class for the Swin Transformer. Compiles it all together and 
    '''
    def __init__(self):
        super().__init__()
        self.Embedding = SwinTEmbedding()
        self.PatchMerging = nn.ModuleList()
        emb_size = 96
        num_class = 5

        for i in range(3):
            self.PatchMerging.append(PatchMerging(emb_size))
            emb_size *= 2
        
        self.stage_1 = SwinEncoder(96, 3)
        self.stage_2 = SwinEncoder(192, 6)
        self.stage_3 = nn.ModuleList([SwinEncoder(384, 12), SwinEncoder(384, 12), SwinEncoder(384, 12)])
        self.stage_4 = SwinEncoder(768, 24)
        
        self.avgpool1d = nn.AdaptiveAvgPool1d(output_size = 1)
        self.avg_pool_layer = nn.AvgPool1d(kernel_size=49)
        
        self.layer = nn.Linear(768, num_class)

    def forward(self, value):
        value = self.Embedding(value)
        value = self.stage_1(value)
        value = self.PatchMerging[0](value)
        value = self.stage_2(value)
        value = self.PatchMerging[1](value)

        for stage in self.stage_3:
            x = stage(value)

        value = self.PatchMerging[2](value)
        value = self.stage_4(value)
        value = self.layer(self.avgpool1d(value.transpose(1, 2)).squeeze(2))

        return x