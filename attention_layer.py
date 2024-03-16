# attention_layer for resnet model defined as in the paper 

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

use_cuda=torch.cuda.is_available()
device=torch.device("cuda" if use_cuda else "cpu")

class Attention_layer(nn.Module):
    def __init__(self,in_channels,kernel_size=7,num_heads=8,image_size=224,inference=False):
        # Kernel Size is nothig but an spatial extent "k" which is described in the paper 
        # As in practice we devide the attention into multiple groups or heads which is the "num_heads" parameter
        super(Attention_layer,self).__init__()
        self.kernel_size=kernel_size 
        self.num_heads=num_heads
        self.image_size=image_size
        self.dk=self.dv=in_channels 
        self.dkh=self.dk//self.num_heads
        self.dvh=self.dv//self.num_heads # These are for defining the weighted matrices Wk,Wv,Wq 

        assert self.dk % self.num_heads==0, "dk remainder should be zero, 40, 8"
        assert self.dv % self.num_heads==0, "dv remainder should be zero, 24, 8"

        self.q_conv=nn.Conv2d(self.dk,self.dk,kernel_size=1).to(device) # Query Vector
        self.k_conv=nn.Conv2d(self.dk,self.dk,kernel_size=1).to(device) # Key Vector
        self.v_conv=nn.Conv2d(self.dv,self.dv,kernel_size=1).to(device) # Value Vector 

        # Relative positional encodings 
        self.h_pos_embed=nn.Parameter(torch.randn(self.dk//2,self.kernel_size,1),requires_grad=True)
        self.w_pos_embed=nn.Parameter(torch.randn(self.dk//2,1,self.kernel_size),requires_grad=True)

        # need to testify this in future 
        self.inference=inference
        if self.inference:
            self.register_parameter('weights',None)
    
    def forward(self,x):
        batch_size,_,height,width=x.size()
        # we need to pad the x as there are upcoming unfolding and folding process, we need to maintain the shape of x need to consistent 
        padded_x = F.pad(x, [(self.kernel_size-1)//2, (self.kernel_size-1)-((self.kernel_size-1)//2), (self.kernel_size-1)//2, (self.kernel_size-1)-((self.kernel_size-1)//2)])
        k=self.k_conv(padded_x)
        q=self.q_conv(x) 
        v=self.v_conv(padded_x)

        # unfold the patches into the [batch_size,num_heads*depth,horizantal_patches,vertical_patches,kerenel_size,kernel_size]
        k=k.unfold(2,self.kernel_size,1).unfold(3,self.kernel_size,1) 
        # print("-----unfolding-k-----",k)
        v=v.unfold(2,self.kernel_size,1).unfold(3,self.kernel_size,1)
        # print("-----unfolding v-------",v)
        # reshape into [batch_size, num_heads, horizontal_patches, vertical_patches, depth_per_head, kernel_size*kernel_size]
        k=k.reshape(batch_size,self.num_heads,height,width,self.dkh,-1)
        v=v.reshape(batch_size,self.num_heads,height,width,self.dvh,-1)

        # reshape into [batch_size,num_heads,height,width,depth_per_head,1]
        q=q.reshape(batch_size,self.num_heads,height,width,self.dkh,1)

        new_var = qk=torch.matmul(q.transpose(4,5),k)
        qk = qk.reshape(batch_size, self.num_heads, height, width, self.kernel_size, self.kernel_size)

        # adding positional encoding 
        qr_h=torch.einsum('bhxydz,cij->bhxyij',q,self.h_pos_embed) 
        qr_w=torch.einsum('bhxydz,cij->bhxyij',q,self.w_pos_embed)

        qk+=qr_h
        qk+=qr_w

        qk=qk.reshape(batch_size,self.num_heads,height,width,1,self.kernel_size*self.kernel_size)
        weights=F.softmax(qk,dim=-1)

        if self.inference:
            self.weights=nn.Parameter(weights)
        
        att_out=torch.matmul(weights,v.transpose(4,5))
        att_out=att_out.reshape(batch_size,-1,height,width)
        return att_out
# killer 