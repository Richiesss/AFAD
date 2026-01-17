# Flowerシミュレーション修正の確認

## 変更内容
**[scripts/run_experiment.py](file:///home/AFAD/scripts/run_experiment.py)**
- Rayなどのマルチプロセス環境でモデル登録が正しく行われるよう、`client_fn` 関数の内部にモデルモジュールのインポート処理を移動しました。

## 検証結果
### 自動テスト
- `source venv/bin/activate && python scripts/run_experiment.py` を実行。
- 修正前は `ValueError: Model 'vit_tiny' not found in registry.` でクラッシュしていましたが、修正後はシミュレーションが開始され、クライアントがモデルをロードする段階（パラメータ不一致の警告が出る箇所）まで進行することを確認しました。

> [!NOTE]
> シミュレーション開始後に `size mismatch` の警告が表示されますが、これは異種モデル（Heterogeneous models）間で初期パラメータ（ResNet18）を共有しているためであり、モデルが見つからないというエラー自体は解決しています。
