import sys,time,argparse
from pathlib import Path
import numpy as np, torch, torchaudio, soundfile as sf, axengine
ROOT=Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT))
T=128; F_HR=432; HOP=512; NFFT=1024; SR=48000
WIN=torch.hann_window(NFFT)
def stft(x):
 X=torch.stft(x.reshape(1,-1),NFFT,HOP,window=WIN,center=True,return_complex=True)
 A=X+1e-4; X=A.abs()**0.2*torch.exp(1j*A.angle())
 r=torch.view_as_real(X).permute(0,3,1,2); return r[:,:,:-1,:]
def istft(sp):
 sp=torch.nn.functional.pad(sp,[0,0,0,1],value=0).permute(0,2,3,1).contiguous()
 X=torch.view_as_complex(sp); X=X.abs()**5.0*torch.exp(1j*X.angle())
 return torch.istft(X,NFFT,HOP,window=WIN,center=True,length=245760)
def main():
 p=argparse.ArgumentParser(); p.add_argument('input'); p.add_argument('output'); p.add_argument('--models',default=str(ROOT/'models')); p.add_argument('--seed',type=int,default=42); p.add_argument('--overlap',type=int,default=64); p.add_argument('--sr',type=int,default=8); a=p.parse_args()
 ce=axengine.InferenceSession(str(Path(a.models)/f'universr_split_{a.sr}k_condenc.axmodel'))
 mc=axengine.InferenceSession(str(Path(a.models)/f'universr_split_{a.sr}k_main_cond_actual.axmodel'))
 mu=axengine.InferenceSession(str(Path(a.models)/f'universr_split_{a.sr}k_main_uncond.axmodel'))
 x,sr=sf.read(a.input); x=torch.tensor(x,dtype=torch.float32).reshape(1,-1)
 x=torchaudio.functional.resample(x,sr,SR); x=torch.nn.functional.pad(x,(0,max(0,245760-x.shape[-1])))[...,:245760]
 Y=stft(x.unsqueeze(1)); total=Y.shape[-1]; lr=80 if a.sr==8 else 170; hf=512-432
 pad=(T-total%T)%T
 Yp=torch.nn.functional.pad(Y,(0,pad),mode='constant') if pad else Y
 total=Yp.shape[-1]
 starts=list(range(0,total-T+1,T-a.overlap))
 spec=torch.zeros_like(Yp); wsum=torch.zeros(total)
 rng=np.random.RandomState(a.seed); t0=time.time()
 for s in starts:
  y=Yp[:,:,:lr,s:s+T].numpy().astype(np.float32)
  z=ce.run(None,{'y':y})[0].astype(np.float32)  # 1,384,1,T
  zz=rng.randn(1,2,F_HR,T).astype(np.float32)
  n=128
  for i in range(4):
   t=np.array([i/4],np.float32); dt=0.25
   def f(q,tt):
    vu=mu.run(None,{'x':q,'t':tt})[0]; vc=mc.run(None,{'x':q,'t':tt,'y_cond':z})[0]
    return (1-1.5)*vu+1.5*vc
   k1=f(zz,t); zm=zz+0.5*dt*k1; k2=f(zm,np.array([i/4+0.125],np.float32)); zz=zz+dt*k2
  cut=max(0,lr-hf); full=np.concatenate([y,zz[:,:,cut:,:]],axis=2) # 1,2,512,T?
  full=torch.from_numpy(full)
  w=torch.ones(T)
  if s>0: w[:a.overlap]=torch.linspace(0,1,a.overlap)
  if s+T>total: w[-(s+T-total):]=0
  spec[:,:,:,s:s+T]+=full*w; wsum[s:s+T]+=w
 spec=spec/wsum.clamp_min(1e-8); out=istft(spec).squeeze()[:245760]
 sf.write(a.output,out.numpy(),SR,subtype='PCM_16'); print('time',time.time()-t0,'shape',out.shape)
if __name__=='__main__': main()
