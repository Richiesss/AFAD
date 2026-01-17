# 進捗表示追加の確認

## 変更内容
**[src/client/afad_client.py](file:///home/AFAD/src/client/afad_client.py)**
- データロードと学習のループに `tqdm` を適用し、各クライアントの学習進捗（エポック数、バッチ進捗、損失値）をコンソールに表示するようにしました。

##検証結果
### 自動テスト
- `source venv/bin/activate && python scripts/run_experiment.py` を実行。
- 以下のように、クライアントごとの進捗バーが表示されることを確認しました。
  ```text
  Client 1 Epoch 1/5:   2%|▏         | 4/188 [00:00<00:14, 12.87batch/s, loss=2.3]
  ```
- これにより、「処理が止まっているのか進んでいるのかわからない」という問題が解消されました。
