# 進捗表示の追加実装計画

## 目標
シミュレーション実行中の進捗が不明瞭であるため、`tqdm` によるプログレスバーとログ出力を追加し、処理が進行していることを視覚的に確認できるようにする。

## 変更内容
### Client
#### [MODIFY] [afad_client.py](file:///home/AFAD/src/client/afad_client.py)
- `tqdm` をインポート。
- `_train` メソッド内のループに `tqdm` を適用し、クライアントIDとエポック数を表示する。

## 検証計画
### 自動テスト
- `source venv/bin/activate && python scripts/run_experiment.py` を実行。
- コンソールにプログレスバーが表示され、学習が進んでいることが確認できるか検証する。
