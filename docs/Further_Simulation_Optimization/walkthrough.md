# データセット削減による高速化確認

## 変更内容
**[src/data/mnist_loader.py](file:///home/AFAD/src/data/mnist_loader.py)**
- `max_samples` 引数を追加し、指定された場合にデータセットをサブセット化（切り詰め）する機能を実装しました。

**[scripts/run_experiment.py](file:///home/AFAD/scripts/run_experiment.py)**
- `load_mnist_data` 呼び出し時に `max_samples=1000` を渡すように変更しました。

## 検証結果
### 自動テスト
- `source venv/bin/activate && python scripts/run_experiment.py` を実行。
- 以下のように、各クライアントの1エポックあたりのバッチ数が **4** （以前は188）になっていることを確認しました。
  ```text
  Client 3 Epoch 1/1: 100%|██████████| 4/4 [00:03<00:00,  1.05batch/s, loss=2.31]
  ```
- これにより、1ラウンドあたりの計算時間が数秒程度に短縮され、快適にデバッグが行えるようになりました。
