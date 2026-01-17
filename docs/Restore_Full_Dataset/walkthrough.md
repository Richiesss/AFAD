# データセット復元の確認

## 変更内容
**[scripts/run_experiment.py](file:///home/AFAD/scripts/run_experiment.py)**
- `load_mnist_data` 呼び出し時の `max_samples=1000` を削除しました。
- これにより、シミュレーションは再びMNISTの全データ（60,000枚）を使用するようになります。

## 補足
- `src/data/mnist_loader.py` の `max_samples` 引数を受け取る機能自体は残しています。将来的に再度デバッグが必要になった際は、`run_experiment.py` に引数を追加するだけで高速化できます。
- シミュレーション設定（`afad_config.yaml`）は `num_rounds: 2`, `local_epochs: 1` のままです。
