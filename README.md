# UniverSR.AXERA

**8 / 16 kHz speech → 48 kHz · AX650 NPU3 · C++ / Python SDK · 6 个 axmodel 共 381 MB**

将 UniverSR 的 UNet 拆成三部分，按 8k/16k 分别编译：

```text
condencoder
main_cond
main_uncond
```

## 结果

| 输入 | 板端 vs PyTorch |
|---|---:|
| 8k → 48k | **0.99193** |
| 16k → 48k | **0.99350** |

C++ 5.12 s 输入约 22–25 s，RTF 4.3–4.9。

## 工程结构

```text
UniverSR.AXERA/
├── model_convert/       # ONNX 拆分 / 校准 / Pulsar2 配置
├── python/              # 板端 Python SDK 和排查脚本
├── cpp/                 # C++ 源码 + 交叉编译脚本
├── samples/             # 中英文示例
├── models/              # 仅 README，axmodel 从 HuggingFace 下载
└── requirements.txt
```

## C++ 构建

```bash
bash cpp/build_ax650.sh
```

## 说明

- 8k/16k 的 `lr_bins` 分别为 80/170，不能混用。
- `main_cond` 必须用实际 cond encoder 输出校准，随机 `y_cond` 会让 `vc` 从 0.9999 掉到 0.9906。
- C++ 和 Python 都使用 T=128、overlap=64、ODE 4 步、CFG 1.5。
- 当前版本没有 12k/24k 模型。

## 数据来源

- 上游：https://github.com/woongzip1/UniverSR
- 示例：Open Speech Repository、公开 16 kHz 人声样例。
