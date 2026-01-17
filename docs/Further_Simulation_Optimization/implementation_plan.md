# データセットサイズ削減による高速化計画

## 目標
シミュレーションの実行速度を劇的に向上させるため、MNISTデータセット全体（60,000枚）ではなく、ごく一部（例：1,000枚）のみを使用して学習を行うオプションを追加する。

## 変更内容
### Data Loader
#### [MODIFY] [mnist_loader.py](file:///home/AFAD/src/data/mnist_loader.py)
- `load_mnist_data` 関数に `max_samples` 引数を追加（デフォルト `None`）。
- `max_samples` が指定された場合、`Subset` を使用してデータセットを切り詰める。

### Scripts
#### [MODIFY] [run_experiment.py](file:///home/AFAD/scripts/run_experiment.py)
- `load_mnist_data` 呼び出し時に `max_samples=1000` などを渡すように変更（デバッグ用）。

## 検証計画
### 自動テスト
- `source venv/bin/activate && python scripts/run_experiment.py` を実行。
- コンソールのプログレスバーで、バッチ総数が減っていること（例: 188 -> 3 程度）を確認し、数秒でラウンドが完了することを検証する。
