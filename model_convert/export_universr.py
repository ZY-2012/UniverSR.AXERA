import os,sys,time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, yaml
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from universr.models.unet import ConvNeXtUNetCond

CFG=ROOT/'ckpts/universr-speech/config.yaml'; CKPT=ROOT/'ckpts/universr-speech/pytorch_model.bin'
cfg=yaml.safe_load(open(CFG)); model=ConvNeXtUNetCond(**cfg['model'])
model.load_state_dict(torch.load(CKPT,map_location='cpu',weights_only=True)); model.eval()
class MeanFreq(nn.Module):
    def forward(self,x): return x.mean(dim=2,keepdim=True)
model.conditioning_encoder.freq_pool=MeanFreq()

class Cond(nn.Module):
    def __init__(self,m,sr): super().__init__(); self.m=m; self.sr=sr
    def forward(self,x,t,y): return self.m(x,t,y,[self.sr])
class Uncond(nn.Module):
    def __init__(self,m,sr): super().__init__(); self.m=m; self.sr=sr
    def forward(self,x,t): return self.m(x,t,None,[self.sr])

# fixed 5.12 s @ 48 kHz: STFT center=True, hop=512 => 481 frames
import os
T=int(os.environ.get("UNIVERSR_T","481")); F_HR=432
for sr,lr in [(8,80),(16,170)]:
    x=torch.randn(1,2,F_HR,T); y=torch.randn(1,2,lr,T); t=torch.rand(1)
    for kind,mod in [('cond',Cond(model,sr)),('uncond',Uncond(model,sr))]:
        mod.eval(); p=ROOT/'model_convert'/f'universr_{sr}k_{kind}.onnx'
        args=(x,t,y) if kind=='cond' else (x,t)
        names=['x','t','y'] if kind=='cond' else ['x','t']
        with torch.no_grad(): out=mod(*args)
        torch.onnx.export(mod,args,str(p),input_names=names,output_names=['v'],opset_version=17,do_constant_folding=True,dynamo=False)
        print(kind,sr,tuple(out.shape),p.stat().st_size/1e6)
