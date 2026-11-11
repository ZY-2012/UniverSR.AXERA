#pragma once
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <numeric>
#include <vector>
#include "third_party/kissfft/kiss_fftr.h"
namespace univ {
constexpr int SR=48000, NFFT=1024, HOP=512, FBINS=513, KEEP=512, TCH=128, FHR=432;
constexpr double PI=3.14159265358979323846;
inline std::vector<float> hann(int n){std::vector<float>w(n);for(int i=0;i<n;++i)w[i]=0.5f-0.5f*(float)std::cos(2.0*PI*i/n);return w;}
class FFT{public:explicit FFT(int n):n_(n){f_=kiss_fftr_alloc(n,0,nullptr,nullptr);i_=kiss_fftr_alloc(n,1,nullptr,nullptr);}~FFT(){free(f_);free(i_);}void forward(const float*x,std::vector<std::complex<float>>&o){o.resize(n_/2+1);kiss_fftr(f_,x,reinterpret_cast<kiss_fft_cpx*>(o.data()));}std::vector<float> inverse(const std::vector<std::complex<float>>&s){std::vector<float>y(n_);kiss_fftri(i_,reinterpret_cast<const kiss_fft_cpx*>(s.data()),y.data());for(auto&v:y)v/=n_;return y;}private:int n_;kiss_fftr_cfg f_=nullptr,i_=nullptr;};
inline std::vector<float> reflect_pad(const std::vector<float>&x,int l,int r){int n=x.size();std::vector<float>o(n+l+r);for(int i=0;i<l;++i)o[i]=x[std::min(l-i,n-1)];std::copy(x.begin(),x.end(),o.begin()+l);for(int i=0;i<r;++i)o[l+n+i]=x[std::max(0,n-2-i)];return o;}
// stft: compressed complex spec, layout [freq][time], 512 bins, no Nyquist
inline std::vector<std::complex<float>> stft(const std::vector<float>&x){auto xp=reflect_pad(x,512,512);int frames=(int)(xp.size()-NFFT)/HOP+1;auto win=hann(NFFT);std::vector<std::complex<float>>sp((size_t)KEEP*frames);FFT fft(NFFT);std::vector<float>fr(NFFT);std::vector<std::complex<float>>s;for(int t=0;t<frames;++t){for(int i=0;i<NFFT;++i)fr[i]=xp[t*HOP+i]*win[i];fft.forward(fr.data(),s);for(int k=0;k<KEEP;++k){auto z=s[k]+std::complex<float>(1e-4f,0);float mag=std::abs(z),ph=std::arg(z);sp[(size_t)k*frames+t]=std::polar(std::pow(mag,0.2f),ph);}}return sp;}
inline std::vector<float> istft(const std::vector<std::complex<float>>&sp,int length){int frames=(int)sp.size()/KEEP;auto win=hann(NFFT);int total=(frames-1)*HOP+NFFT;std::vector<float>y(total,0),env(total,0),all((size_t)frames*NFFT);FFT fft(NFFT);for(int t=0;t<frames;++t){std::vector<std::complex<float>>s(FBINS);for(int k=0;k<KEEP;++k){auto z=sp[(size_t)k*frames+t];s[k]=std::polar(std::pow(std::abs(z),5.0f),std::arg(z));}s[FBINS-1]=0;auto f=fft.inverse(s);for(int i=0;i<NFFT;++i)all[(size_t)t*NFFT+i]=f[i];}for(int t=0;t<frames;++t){int off=t*HOP;for(int i=0;i<NFFT;++i){y[off+i]+=all[(size_t)t*NFFT+i]*win[i];env[off+i]+=win[i]*win[i];}}y=std::vector<float>(y.begin()+512,y.end()-512);env=std::vector<float>(env.begin()+512,env.end()-512);for(size_t i=0;i<y.size();++i)y[i]/=std::max(env[i],1e-12f);if((int)y.size()<length)y.resize(length,0);y.resize(length);return y;}
inline std::vector<float> resample(const std::vector<float>&x,int orig,int target){if(orig==target)return x;int g=std::gcd(orig,target);int o=orig/g,n=target/g;double roll=0.99,ws=6.0,base=std::min(o,n)*roll;int width=(int)std::ceil(ws*o/base),ks=2*width+o;std::vector<float>kern((size_t)n*ks);for(int p=0;p<n;++p)for(int i=0;i<ks;++i){double t=((double)(i-width)/o)-((double)p/n);t=std::max(-ws,std::min(ws,t*base));double w=std::pow(std::cos(t*PI/ws/2.0),2.0);double sinc=std::abs(t)<1e-12?1.0:std::sin(PI*t)/(PI*t);kern[(size_t)p*ks+i]=(float)(sinc*w*base/o);}std::vector<float>pad(width+x.size()+width+o,0);std::copy(x.begin(),x.end(),pad.begin()+width);int nout=((int)pad.size()-ks)/o+1;std::vector<float>y((size_t)nout*n);for(int p=0;p<n;++p)for(int j=0;j<nout;++j){float a=0;int off=j*o;for(int i=0;i<ks;++i)a+=pad[off+i]*kern[(size_t)p*ks+i];y[(size_t)j*n+p]=a;}y.resize(std::min<int>(y.size(),(int)std::ceil((double)n*x.size()/o)));return y;}
}
