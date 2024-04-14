from bitnet import BitLinear
import math 
import torch 
import torch.nn as nn 
import torch.utils.checkpoint as checkpoint 
from einops import rearrange 
from timm.models.layers import DropPath,to_2tuple,trunc_normal_ 

class WindowAttention(nn.Module):
    r"""
    Window Based multi-head self attention (W-Msa) module with relative position bias. 
    It suppourts both of shifted and non-shifted window.

    Args: 
    dim(int): No of input channels 
    window size(tuple[int]): The height and width of the window.
    num_heads(int):No of attention Heads
    qkv_bias(bool,optional): If True, add a learnable bias to query,key,value Default: True. 
    qk_scale(Float|None,Optional): Override default qk scale of head_dim** -0.5 if set.
    attn_drop(float,optional):  Dropout ratio of attention weight : Default: 0.0 
    proj_drop(float,optional): Dropout ratio of output. Default :0.0 
    """

    def __init__(self,dim,window_size,num_heads,qkv_bias=True,qk_scale=None,attn_drop=0.,proj_drop=0.):
        super().__init__()
        self.dim=dim 
        self.window_size=window_size # wh,Ww
        self.num_heads=num_heads
        head_dim=dim//num_heads
        self.scale=qk_scale or head_dim ** -0.5
        
        # define a parameter table of relative position bias 
        self.relative_position_bias_table=nn.Parameter(
            torch.zeros((2*window_size[0]-1)*(2*window_size[1]-1),num_heads))
        
        # get pair-wise relative position index for each token inside the window
        
        coords_h=torch.arange(self.window_size[0])
        coords_w=torch.arange(self.window_size[1])
        coords=torch.stack(torch.meshgird([coords_h,coords_w])) # 2, Wh,Ww
        coords_flatten=torch.flatten(coords,1)
        relative_coords=coords_flatten[:,:,None]-coords_flatten[:,None,:] # 2, Wh*Ww, Wh*Ww
        relative_coords=relative_coords.permute(1,2,0).contiguous() # Wh*Ww,Wh*Ww,2 
        relative_coords[:,:,0]+=self.window_size[0]-1 # shift to start from 0 
        relative_coords[:,:,1]+=self.window_size[1]-1
        relative_coords[:,:,0]*=2*self.window_size[1]-1 
        relative_position_index=relative_coords.sum(-1)
        self.register_buffer("relative_position_index",relative_position_index)

        self.qkv=BitLinear(dim,dim*3,bias=qkv_bias)
        self.attn_drop=nn.Dropout(attn_drop)
        self.proj=BitLinear(dim,dim)
        self.proj_drop=nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table,std=.02)
        self.softmax=nn.Softmax(dim=-1)
    
    def forward(self,x,mask=None):
        """
        Args: 
        x: input features with shape of (num_windows*B,N,C)
        mask: (0/-inf) mask with shape of (num_windows,Wh*Ww,Wh*Ww) or None
        """
        B_,N,C= x.shape
        qkv=self.qkv(x).reshape(B_,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4)
        q,k,v=qkv[0],qkv[1],qkv[2]

        q=q*self.scale
        attn=(q@k.transpose(-2,-1))

        relative_position_bias=self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0]*self.window_size[1],self.window_size[0]*self.window_size[1],-1 # Wh*Ww,Wh*Ww,nH
        )
        relative_position_bias=relative_position_bias.permute(2,0,1).contiguous() # nH,Wh*Ww,Wh*Ww
        attn=attn+relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW=mask.shape[0]
            attn=attn.view(B_//nW,nW,self.num_heads,N,N)+mask.unsqueeze(1).unsqueeze(0)
            attn=attn.view(-1,self.num_heads,N,N)
            attn=self.softmax(attn)
        else:
            attn=self.softmax(attn)
        
        attn=self.attn_drop(attn)
        x=(attn@v).transpose(1,2).reshape(B_,N,C)
        x=self.proj(x)
        x=self.proj_drop(x)
        return x
    
    def extra_repr(self) -> str:
        return f'dim={self.dim},window_size={self.window_size},num_heads={self.num_heads}'
    
    def flops(self,N):
        # cal flops for 1 window with token lenght of N
        flops=0
        flops+=N*self.dim*3*self.dim
        flops+=self.num_heads*N*(self.dim//self.num_heads)*N
        flops+=self.num_heads*N*N*(self.dim//self.num_heads)
        flops+=N*self.dim*self.dim

        return flops 
# print("killer")