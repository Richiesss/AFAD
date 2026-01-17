# シミュレーション設定の最適化計画

## 目標
シミュレーションの実行時間が長すぎるため、パラメータを調整してデバッグサイクルを短縮する。

## 変更内容
### Config
#### [MODIFY] [afad_config.yaml](file:///home/AFAD/config/afad_config.yaml)
- `experiment.num_rounds`: 50 -> 2
- `training.local_epochs`: 5 -> 1

この変更により、全体の計算量が大幅に削減され、数分で動作確認が完了するようになります。

## 検証計画
### 自動テスト
- `source venv/bin/activate && python scripts/run_experiment.py` を実行。
- 処理が数分以内に完了し、ラウンドが進むことを確認する。
