# データセット復元計画

## 目標
シミュレーションで使用するデータセットを、縮小版（1000サンプル）から元のフルセット（60000サンプル）に戻す。

## 変更内容
### Scripts
#### [MODIFY] [run_experiment.py](file:///home/AFAD/scripts/run_experiment.py)
- `load_mnist_data` の呼び出しから `max_samples=1000` を削除する。

注: `src/data/mnist_loader.py` の `max_samples` 引数を受け取る機能自体は、将来のデバッグ用に残しておく（デフォルトが `None` なので影響しない）。

## 検証計画
### 手動確認
- コード修正後、`scripts/run_experiment.py` を閲覧し、引数が削除されていることを確認する。
- ユーザー指示により、実行確認まで行うかは任意だが、実行すると時間がかかるためコード修正のみを主とする。
