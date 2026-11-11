import sys, numpy as np, torch, yaml, soundfile as sf
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from universr.models.unet import ConvNeXtUNetCond
cfg=yaml.safe_load(open('ckpts/universr-speech/config.yaml')); m=ConvNeXtUNetCond(**cfg['model']); m.load_state_dict(torch.load('ckpts/universr-speech/pytorch_model.bin',map_location='cpu',weights_only=True)); m.eval()
rng=np.random.RandomState(123); T=128
x=torch.from_numpy(rng.randn(1,2,432,T).astype(np.float32)); y=torch.from_numpy(rng.randn(1,2,80,T).astype(np.float32)); t=torch.tensor([0.0])
np.savez('build/chunk_input.npz',x=x.numpy(),y=y.numpy(),t=t.numpy())
pe=m.freq_pos_enc()[:80,:]; sr=m.sr_embedder(torch.tensor([m.sr_to_idx[8]])).expand(1,-1)
with torch.no_grad():
 z=m.conditioning_encoder(y,pe,sr).unsqueeze(2); vc=m(x,t,y,[8]); vu=m(x,t,None,[8])
np.savez('build/chunk_host.npz',z=z.numpy(),vc=vc.numpy(),vu=vu.numpy()); print('host',z.shape,vc.shape,vu.shape)
