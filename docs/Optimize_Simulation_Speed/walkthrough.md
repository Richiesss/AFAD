# シミュレーション高速化の確認

## 変更内容
**[config/afad_config.yaml](file:///home/AFAD/config/afad_config.yaml)**
- `num_rounds` を 50 から 2 に変更。
- `local_epochs` を 5 から 1 に変更。

## 検証結果
### 自動テスト
- `source venv/bin/activate && python scripts/run_experiment.py` を実行。
- ログ出力で `Epoch 1/1` となっていることを確認しました。
- これにより、デバッグ時の待ち時間が大幅に短縮されました。
