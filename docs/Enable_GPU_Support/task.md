# GPUサポートの有効化

- [x] 現状のGPU実装状況を確認する (`scripts/run_experiment.py`, `src/client/afad_client.py`) <!-- id: 0 -->
- [x] `scripts/run_experiment.py` で `start_simulation` に `client_resources` (num_gpus) を設定する <!-- id: 1 -->
- [x] `src/client/afad_client.py` でモデルとデータが正しくデバイス(`cuda`)に転送されているか確認・修正する <!-- id: 2 -->
- [x] 動作確認を行う（GPUが利用可能であれば使用されることを確認） <!-- id: 3 -->
