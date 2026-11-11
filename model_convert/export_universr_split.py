import os,sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, yaml
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from universr.models.unet import ConvNeXtUNetCond
from einops import rearrange
T=int(os.environ.get('UNIVERSR_T','128')); F_HR=432
cfg=yaml.safe_load(open(ROOT/'ckpts/universr-speech/config.yaml')); m=ConvNeXtUNetCond(**cfg['model'])
m.load_state_dict(torch.load(ROOT/'ckpts/universr-speech/pytorch_model.bin',map_location='cpu',weights_only=True)); m.eval()
class MeanFreq(nn.Module):
 def forward(self,x): return x.mean(dim=2,keepdim=True)
m.conditioning_encoder.freq_pool=MeanFreq()
class CondEnc(nn.Module):
 def __init__(self,m,sr): super().__init__(); self.m=m; self.sr=sr; self.idx=m.sr_to_idx[sr]
 def forward(self,y):
  lr=self.m.sr_to_lr_bins[self.sr]; pe=self.m.freq_pos_enc()[:lr,:]
  sr_emb=self.m.sr_embedder(torch.tensor([self.idx],dtype=torch.long)).expand(y.shape[0],-1)
  return self.m.conditioning_encoder(y,pe,sr_emb).unsqueeze(2)
class Main(nn.Module):
 def __init__(self,m,sr,cond): super().__init__(); self.m=m; self.sr=sr; self.cond=cond
 def forward(self,x,t,y_cond=None):
  x,pad=self.m._pad_frames(x)
  if y_cond is not None and pad: y_cond=torch.nn.functional.pad(y_cond,[0,pad,0,0],mode='reflect')
  B,_,F,TT=x.shape
  t_embed=self.m.time_embedder(t)
  sr_emb=self.m.sr_embedder(torch.tensor([self.m.sr_to_idx[self.sr]],dtype=torch.long)).expand(B,-1)
  t_embed=t_embed+self.m.sr_projector(sr_emb)
  if y_cond is None:
   y_cond=self.m.uncond_emb.reshape(1,self.m.cond_dim,1,1).expand(B,self.m.cond_dim,1,TT)
  pe=self.m.freq_pos_enc(); hf_start=self.m.total_freq_bins-self.m.hr_freq_bins
  pe_high=pe[hf_start:,:]
  film=self.m.film_generator(pe_high); gamma,beta=torch.chunk(film,2,dim=-1)
  gamma=rearrange(gamma,'f d -> 1 d f 1'); beta=rearrange(beta,'f d -> 1 d f 1')
  spatial=y_cond*gamma+beta
  x=self.m.init_conv(torch.cat([x,spatial],1)); skips=[x]
  for e in self.m.encoders: x=e(x,t_embed); skips.append(x)
  x=self.m.midcoder(x,t_embed)
  for d in self.m.decoders:
   sk=skips.pop()
   if x.shape!=sk.shape: x=torch.nn.functional.interpolate(x,size=sk.shape[2:])
   x=x+sk; x=d(x,t_embed)
  sk=skips.pop(); x=x+sk; x=self.m.final_conv(x)
  if pad: x=x[...,:-pad]
  return x
for sr,lr in [(8,80),(16,170)]:
 y=torch.randn(1,2,lr,T); x=torch.randn(1,2,F_HR,T); t=torch.rand(1)
 ce=CondEnc(m,sr).eval(); mc=Main(m,sr,True).eval(); mu=Main(m,sr,False).eval()
 with torch.no_grad():
  z=ce(y)
  print('z',z.shape,'vcond',mc(x,t,z).shape,'vuncond',mu(x,t).shape)
  for name,mod,args,names in [('condenc',ce,(y,),['y']),('main_cond',mc,(x,t,z),['x','t','y_cond']),('main_uncond',mu,(x,t),['x','t'])]:
   p=ROOT/'model_convert'/f'universr_split_{sr}k_{name}.onnx'
   torch.onnx.export(mod,args,str(p),input_names=names,output_names=['v'],opset_version=17,do_constant_folding=True,dynamo=False)
   print(name,sr,p.stat().st_size/1e6)
