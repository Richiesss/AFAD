# GPUサポート有効化計画

## 目標
シミュレーションでGPU（RTX 4090）を有効活用し、計算速度を向上させる。

## 現状
- `scripts/run_experiment.py` で `CLIENT_RESOURCES = {"num_cpus": 1.0, "num_gpus": 0.0}` と設定されており、GPU使用が無効化されている。
- 環境にはRTX 4090が存在する。

## 変更内容
### Scripts
#### [MODIFY] [run_experiment.py](file:///home/AFAD/scripts/run_experiment.py)
- `CLIENT_RESOURCES` の定義を動的に変更する。
- GPUが利用可能な場合、`num_gpus` を `0.15` (約6並列可能) に設定する。GPUがない場合は `0.0` のまま。

```python
# 例
device_type = "cuda" if torch.cuda.is_available() else "cpu"
num_gpus = 0.15 if device_type == "cuda" else 0.0
CLIENT_RESOURCES = {"num_cpus": 1.0, "num_gpus": num_gpus}
```

## 検証計画
### 自動テスト
- `source venv/bin/activate && python scripts/run_experiment.py` を実行。
- ログ出力や実行速度を確認する。
- 可能であれば、実行中に `nvidia-smi` でプロセスがGPUを使っているか確認する（今回はツール経由でそこまでリアルタイム確認は難しいが、ログに進捗が出ればOK）。
