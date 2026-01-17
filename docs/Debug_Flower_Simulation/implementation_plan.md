# Rayワーカーでのモデル登録漏れの修正

## 目標
Flowerシミュレーション（Rayバックエンド）実行時に発生する `ValueError: Model 'vit_tiny' not found in registry.` エラーを修正する。
この問題は、メインスクリプトでのインポート時に行われるデコレータによるモデル登録が、`client_fn`を実行するRayのワーカープロセスに引き継がれないために発生しています。

## 変更内容
### Scripts
#### [MODIFY] [run_experiment.py](file:///home/AFAD/scripts/run_experiment.py)
- モデルモジュールのインポートを `client_fn`（ネストされた関数）の内部に移動（または複製）します。これにより、関数がリモートワーカーで実行される際にインポートが実行され、そのワーカーの `ModelRegistry` にモデルが登録されるようになります。

## 検証計画
### 自動テスト
- 実験スクリプトを実行し、特定のエラーなしでシミュレーションが開始されるか確認します。
  ```bash
  source venv/bin/activate && python scripts/run_experiment.py
  ```
- ログで "Starting Flower simulation" が表示され、最初のクライアント作成が成功することを確認します。
