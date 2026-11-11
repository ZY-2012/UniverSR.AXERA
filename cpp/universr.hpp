#pragma once
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <fstream>
#include <string>
#include <vector>
#include "dsp.hpp"
#include "model_runner.hpp"
namespace univ {
class UniverSR {
public:
 bool load(const std::string& dir,int sr){sr_=sr; lr_=(sr==8?80:170); cut_=std::max(0,lr_-80); auto P=[&](const char*n){return dir+"/universr_split_"+std::to_string(sr)+"k_"+n+".axmodel";};
  return ce_.load(P("condenc"))&&mc_.load(P("main_cond_actual"))&&mu_.load(P("main_uncond"));}
 std::vector<float> enhance(const std::vector<float>& wav,int sr){
  auto x=resample(wav,sr,SR); x.resize(245760,0.f); auto Y=stft(x); int total=245760/HOP+1; if((int)Y.size()/KEEP<total) total=(int)Y.size()/KEEP;
  int totalP=((total+TCH-1)/TCH)*TCH; std::vector<std::complex<float>>Yp((size_t)KEEP*totalP); for(int f=0;f<KEEP;++f)for(int t=0;t<total;++t)Yp[(size_t)f*totalP+t]=Y[(size_t)f*total+t];
  if(std::getenv("UNIV_DUMP")){std::ofstream of("Yp.bin",std::ios::binary);for(int fq=0;fq<KEEP;++fq)for(int tq=0;tq<totalP;++tq){float a=Yp[(size_t)fq*totalP+tq].real(),b=Yp[(size_t)fq*totalP+tq].imag();of.write((char*)&a,4);of.write((char*)&b,4);}}
  std::vector<std::complex<float>> S((size_t)KEEP*totalP); std::vector<float> W(totalP,0.f); std::mt19937 gen(42); std::normal_distribution<float> nd(0,1);
  const char* xf=std::getenv("UNIV_X0"); if(xf){std::ifstream f(xf,std::ios::binary);if(f){x0buf_.resize((size_t)7*2*FHR*TCH);f.read((char*)x0buf_.data(),x0buf_.size()*sizeof(float));}}
  int chunk=0;
  for(int s=0;s+TCH<=totalP;s+=64,++chunk){
   std::vector<float> y((size_t)2*lr_*TCH), x0((size_t)2*FHR*TCH);
   for(int c=0;c<2;++c)for(int f=0;f<lr_;++f)for(int t=0;t<TCH;++t)y[((size_t)c*lr_+f)*TCH+t]=(c==0?Yp[(size_t)f*totalP+s+t].real():Yp[(size_t)f*totalP+s+t].imag());
   if(!x0buf_.empty()){size_t off=(size_t)chunk*2*FHR*TCH;std::copy(x0buf_.begin()+off,x0buf_.begin()+off+x0.size(),x0.begin());}else for(auto&v:x0)v=nd(gen);
   std::vector<float> z; if(!ce_.set_input(0,y.data(),y.size())||!ce_.run()||!ce_.get_output(0,z,384*TCH))return {};
   std::vector<float> vel((size_t)2*FHR*TCH),vu,vc;
   auto cfg=[&](const std::vector<float>&q,float t,std::vector<float>&out){float tt=t;std::vector<float>v;
    if(!mu_.set_input(0,q.data(),q.size())||!mu_.set_input(1,&tt,1)||!mu_.run()||!mu_.get_output(0,vu,q.size()))return false;
    if(!mc_.set_input(0,q.data(),q.size())||!mc_.set_input(1,&tt,1)||!mc_.set_input(2,z.data(),z.size())||!mc_.run()||!mc_.get_output(0,vc,q.size()))return false;
    out.resize(q.size());for(size_t i=0;i<q.size();++i)out[i]=(1-1.5f)*vu[i]+1.5f*vc[i];return true;};
   for(int i=0;i<4;++i){float dt=0.25f;std::vector<float>k1,k2,mid(x0.size());if(!cfg(x0,i*0.25f,k1))return {};for(size_t j=0;j<x0.size();++j)mid[j]=x0[j]+0.5f*dt*k1[j];if(!cfg(mid,i*0.25f+0.125f,k2))return {};for(size_t j=0;j<x0.size();++j)x0[j]+=dt*k2[j];}
   for(int t=0;t<TCH;++t){int g=s+t;float w=1.f;if(s>0&&t<64)w=(float)t/64.f;if(s+TCH>total){int rem=s+TCH-total;if(t>=TCH-rem)w=0.f;}if(w>0)W[g]+=w;}
   for(int f=0;f<KEEP;++f){
    for(int t=0;t<TCH;++t){int g=s+t;float w=1.f;if(s>0&&t<64)w=(float)t/64.f;if(s+TCH>total){int rem=s+TCH-total;if(t>=TCH-rem)w=0.f;}float re,im;
     if(f<lr_){re=Yp[(size_t)f*totalP+g].real();im=Yp[(size_t)f*totalP+g].imag();}
     else{int q=f-80;int fi=((0)*FHR+q)*TCH+t;int fq=((1)*FHR+q)*TCH+t;re=x0[fi];im=x0[fq];}
     S[(size_t)f*totalP+g]+=std::complex<float>(re,im)*w;}}
  }
  for(int f=0;f<KEEP;++f)for(int t=0;t<totalP;++t){float w=W[t];if(w>1)S[(size_t)f*totalP+t]/=w;}
  if(std::getenv("UNIV_DUMP_S")){std::ofstream ofs("S.bin",std::ios::binary);for(int fq=0;fq<KEEP;++fq)for(int tq=0;tq<totalP;++tq){float a=S[(size_t)fq*totalP+tq].real(),b=S[(size_t)fq*totalP+tq].imag();ofs.write((char*)&a,4);ofs.write((char*)&b,4);}}
  auto out=istft(S,245760); return out;
 }
private:
 int sr_=8,lr_=80,cut_=0; AxModel ce_,mc_,mu_; std::vector<float> x0buf_;
};
}
