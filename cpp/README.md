# UniverSR.AXERA C++ runtime

AX650 C++17 runtime for the three-way split UniverSR graph:

```text
condencoder -> main_cond / main_uncond -> vocoder-free iSTFT
```

Host DSP: resample, STFT/ISTFT, amplitude compression, 4-step midpoint flow,
CFG 1.5, 128-frame chunks with 64-frame overlap.

## Build

```bash
bash cpp/build_ax650.sh
```

## Run on board

```bash
export LD_LIBRARY_PATH=/soc/lib:$LD_LIBRARY_PATH
./bin/universr_ax650 models samples/listen/OSR_us_000_0010_input_8k_5s.wav out.wav --sr 8
./bin/universr_ax650 models samples/listen/speech_2_input_16k.wav out16.wav --sr 16
```

Models are the six `universr_split_{8,16}k_*.axmodel` files.

The C++ runtime and Python SDK both use 8k/16k -> 48k, 128-frame chunks,
64-frame overlap. C++ has no Python/torch dependency.
