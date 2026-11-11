import json
from pathlib import Path
OUT=Path(__file__).resolve().parent/'configs'; OUT.mkdir(exist_ok=True)
for sr,lr in [(8,80),(16,170)]:
 for kind in ['cond','uncond']:
  ins={'x':(1,2,432,481),'t':(1,)}
  if kind=='cond': ins['y']=(1,2,lr,481)
  cfg={"model_type":"ONNX","target_hardware":"AX650","npu_mode":"NPU3","input":f"/workspace/model_convert/universr_{sr}k_{kind}.onnx","output_dir":f"/workspace/build/compile/universr_{sr}k_{kind}","output_name":"model.axmodel","work_dir":f"/workspace/build/compile/universr_{sr}k_{kind}_work","input_shapes":",".join(f"{k}:{'x'.join(map(str,v))}" for k,v in ins.items()),"onnx_opt":{"disable_onnx_optimization":False,"enable_onnxsim":False,"model_check":True},"quant":{"calibration_method":"MinMax","precision_analysis":True,"precision_analysis_method":"EndToEnd","highest_mix_precision":False,"layer_configs":[{"start_tensor_names":["DEFAULT"],"end_tensor_names":["DEFAULT"],"data_type":"U16","weight_data_type":"S8","output_data_type":"U16"}],"input_configs":[{"tensor_name":k,"calibration_dataset":f"/workspace/build/calib/{sr}k_{kind}/{k}.tar.gz","calibration_format":"Numpy","calibration_size":8,"calibration_mean":[],"calibration_std":[]} for k in ins]},"input_processors":[],"compiler":{"check":0}}
  p=OUT/f'universr_{sr}k_{kind}.json'; p.write_text(json.dumps(cfg,indent=2)+'\n'); print(p)
