import os,sys,json,tarfile
from pathlib import Path
import numpy as np, torch, torchaudio, yaml
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from universr.utils.spectral_ops import AmplitudeCompressedComplexSTFT
cfg=yaml.safe_load(open(ROOT/'ckpts/universr-speech/config.yaml')); tr=AmplitudeCompressedComplexSTFT(**cfg['transform']).eval()
base=ROOT/'build/calib'; base.mkdir(parents=True,exist_ok=True)
def stft(x):
 s=tr(x); r=torch.view_as_real(s.squeeze(1)).permute(0,3,1,2); return r[:,:,:-1,:]
rng=np.random.RandomState(42)
for sr,lr in [(8,80),(16,170)]:
 for kind in ['cond','uncond']:
  d=base/f'{sr}k_{kind}'; d.mkdir(parents=True,exist_ok=True)
  for i in range(8):
   x=np.asarray(rng.randn(1,2,432,481),dtype=np.float32)
   t=np.asarray([rng.rand()],dtype=np.float32)
   np.save(d/f'x_{i:04d}.npy',x); np.save(d/f't_{i:04d}.npy',t)
   if kind=='cond':
    y=np.asarray(rng.randn(1,2,lr,481),dtype=np.float32); np.save(d/f'y_{i:04d}.npy',y)
  for key in (['x','t','y'] if kind=='cond' else ['x','t']):
   with tarfile.open(d/f'{key}.tar.gz','w:gz') as tf:
    for f in sorted(d.glob(f'{key}_*.npy')): tf.add(f,arcname=f.name)
  print(sr,kind,len(list(d.glob('*.npy'))))
