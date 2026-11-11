import sys, numpy as np, torch, yaml
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from universr.models.unet import ConvNeXtUNetCond
cfg=yaml.safe_load(open('ckpts/universr-speech/config.yaml')); m=ConvNeXtUNetCond(**cfg['model']); m.load_state_dict(torch.load('ckpts/universr-speech/pytorch_model.bin',map_location='cpu',weights_only=True)); m.eval()
d=np.load('build/chunk_input.npz'); y=torch.from_numpy(d['y']); rng=np.random.RandomState(777); x0=torch.from_numpy(rng.randn(1,2,432,128).astype(np.float32)); np.save('build/chunk_x0.npy',x0.numpy())
def cfgv(x,t):
 with torch.no_grad(): vu=m(x,torch.tensor([t]),None,[8]); vc=m(x,torch.tensor([t]),y,[8])
 return (1-1.5)*vu+1.5*vc
x=x0.clone(); ts=np.linspace(0,1,5)
with torch.no_grad():
 for i in range(4):
  t=float(ts[i]); dt=float(ts[i+1]-ts[i]); k1=cfgv(x,t); k2=cfgv(x+0.5*dt*k1,t+0.5*dt); x=x+dt*k2
np.save('build/chunk_flow_host.npy',x.numpy()); print(x.shape)
