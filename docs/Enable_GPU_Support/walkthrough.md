# GPUサポート有効化の確認

## 変更内容
**[scripts/run_experiment.py](file:///home/AFAD/scripts/run_experiment.py)**
- `CLIENT_RESOURCES` を修正し、GPU（cuda）が利用可能な場合に `num_gpus: 0.15` を設定するようにしました。
- これにより、1枚のGPUで約6つのクライアントを並列に実行できます。

**[src/client/afad_client.py](file:///home/AFAD/src/client/afad_client.py)**
- デバッグ用に、クライアントが初期化されたデバイス（`cuda` or `cpu`）をログ出力する行を追加しました。

## 検証結果
### 自動テスト
- `source venv/bin/activate && python scripts/run_experiment.py` を実行。
- 以下のログにより、GPUが認識され使用されていることを確認しました。
  ```text
  INFO - Client 3 initialized on device: cuda
  ```
- また、学習速度が CPU時の **約1 batch/s** から **15~30 batch/s** に大幅に向上しました。
