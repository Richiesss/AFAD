# HeteroFL集約エラー(Shape Mismatch)の修正計画

## 目標
`ValueError: operands could not be broadcast together...` エラーを修正する。
これは、グローバルモデルが小さい状態（例: ResNet18）で始まり、後から大きいモデル（例: ResNet50）の更新が来た際に、配列サイズが足りずブロードキャストに失敗するために発生しています。

## 変更内容
### Strategy
#### [MODIFY] [heterofl_aggregator.py](file:///home/AFAD/src/server/strategy/heterofl_aggregator.py)
- `aggregate` メソッドのロジックを一新する。
1. **最大シェイプの特定**: `global_params` と `results` 内の全クライアントのパラメータを走査し、各レイヤーごとの最大次元数を決定する。
2. **アキュムレータの作成**: 最大シェイプを持つ `weighted_sum` と `weights_count` 配列を作成する。
3. **集約**: 各クライアントのパラメータを、最大シェイプの対応するスライス位置に加算（`sum`と`count`）する。
4. **平均化**: `count > 0` の部分を正規化（割り算）し、`count == 0` の部分は元の `global_params`（存在する場合）または0で埋める。

## 検証計画
### 自動テスト
- `source venv/bin/activate && python scripts/run_experiment.py` を実行。
- シミュレーションがクラッシュせずに完了し、ラウンドの集約（`aggregate_fit`）が成功することを確認する。
