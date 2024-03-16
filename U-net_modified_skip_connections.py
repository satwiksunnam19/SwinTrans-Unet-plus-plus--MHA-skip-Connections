import torch
import numpy as np 
import torch.nn as nn 

class EncoderBlock(nn.Module):
    # consists of Conv->Relu->Maxpool 
    def __init__(self,in_chans,out_chans,layers=2,sampling_factor=2,padding="same"):
        super().__init__()
        self.encoder=nn.ModuleList()
        self.encoder.append(nn.Conv2d(in_chans,out_chans,3,1,padding=padding))
        self.encoder.append(nn.ReLU())
        for _ in range(layers-1):
            self.encoder.append(nn.Conv2d(out_chans,out_chans,3,1,padding=padding))
            self.encoder.append(nn.ReLU())
        self.mp=nn.MaxPool2d(sampling_factor)
    
    def forward(self,x):
        for enc in self.encoder:
            x=enc(x)
        mp_out=self.mp(x)
        return mp_out,x
    
# class DecoderBlock(nn.Module):
#     # consists a 2x2 transposed convolution -> Conv-> relu 
#     def __init__(self,in_chans,out_chans,layers=2,skip_connection=True,sampling_factor=2,padding="same"):
#         super().__init__()
#         skip_factor=1 if skip_connection else 2 
#         self.decoder=nn.ModuleList()
#         self.tconv=nn.ConvTranspose2d(in_chans,in_chans//2,sampling_factor,sampling_factor)
#         self.decoder.append(nn.Conv2d(in_chans//skip_factor,out_chans,3,1,padding=padding))
#         self.decoder.append(nn.ReLU())
#         for _ in range(layers-1):
#             self.decoder.append(nn.Conv2d(out_chans,out_chans,3,1,padding=padding))
#             self.decoder.append(nn.ReLU())
        
#     def forward(self,x,enc_features=None):
#         x=self.tconv(x)
#         if self.skip_connection:
#             if self.padding!="same":
#                 #crop the enc_features to the same size as input 
#                 w=x.size(-1)
#                 c=(enc_features.size(-1)-w)//2
#                 enc_features=enc_features[:,:,c:c+w,c:c+w]
#             x=torch.cat((enc_features,x),dim=1)
        
#         for dec in self.decoder:
#             x=dec(x)
#         return x 
    

class UNet(nn.Module):
    def __init__(self, nclass=1, in_chans=1, depth=5, layers=2, sampling_factor=2, skip_connection=True, padding="same"):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        out_chans = 64
        for _ in range(depth):
            self.encoder.append(EncoderBlock(in_chans, out_chans, layers, sampling_factor, padding))
            in_chans = out_chans
            out_chans *= 2

        out_chans = in_chans // 2

        for _ in range(depth - 1):
            self.decoder.append(DecoderBlock(in_chans, out_chans, layers, skip_connection, sampling_factor, padding))
            in_chans = out_chans
            out_chans //= 2

        # Add the 1x1 conv to produce final classes
        self.logits = nn.Conv2d(in_chans, nclass, 1, 1)

    def forward(self, x):
        encoded = []
        for enc in self.encoder:
            x, enc_output = enc(x)
            encoded.append(enc_output)
        x = encoded.pop()

        for dec in self.decoder:
            enc_output = encoded.pop()
            x = dec(x, enc_output)

        return self.logits(x)


# we are going to replace the skip connections working in the decoder block with the MHA

class DecoderBlock(nn.Module):
    # consists a 2x2 transposed convolution -> Conv-> relu 
    def __init__(self,in_chans,out_chans,layers=2,skip_connection=True,sampling_factor=2,padding="same"):
        super().__init__()
        self.skip_connection=skip_connection
        self.padding=padding 
        skip_factor=1 if skip_connection else 2 
        self.decoder=nn.ModuleList()
        self.tconv=nn.ConvTranspose2d(in_chans,in_chans//2,sampling_factor,sampling_factor)
        self.decoder.append(nn.Conv2d(in_chans//skip_factor,out_chans,3,1,padding=padding))
        self.decoder.append(nn.ReLU())
        for _ in range(layers-1):
            self.decoder.append(nn.Conv2d(out_chans,out_chans,3,1,padding=padding))
            self.decoder.append(nn.ReLU())
        if self.skip_connection:
            self.mha = nn.MultiheadAttention(embed_dim=out_chans, num_heads=8, batch_first=True)

    def forward(self,x,enc_features=None):
        x=self.tconv(x)
        if self.skip_connection:
            if self.padding!="same":
                #crop the enc_features to the same size as input 
                w=x.size(-1)
                c=(enc_features.size(-1)-w)//2
                enc_features=enc_features[:,:,c:c+w,c:c+w]
            # reshape for MHA 
            b,c,h,w=x.shape
            x_mha=x.permute(0,2,3,1).reshape(b,h*w,c)
            enc_features_mha=enc_features.permute(0,2,3,1).reshape(b,h*w,c)
            attn_op,_=self.mha(x_mha,enc_features_mha,enc_features_mha)
            attn_op=attn_op.reshape(b,h,w,c).permute(0,3,1,2)
            x=torch.cat((attn_op,x),dim=1)
        
        for dec in self.decoder:
            x=dec(x)
        return x 
    

model = UNet(nclass=2, in_chans=3, depth=4, layers=2, sampling_factor=2, skip_connection=True, padding="same")
dummy_input = torch.randn(2, 3, 256, 256)
output = model(dummy_input)
print("Output shape:", output.shape)
