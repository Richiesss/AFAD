# AFAD (Adaptive Federated Architecture Distribution)

AFADは、連合学習（Federated Learning）におけるモデル異質性問題（Model Heterogeneity）を解決するためのハイブリッドフレームワークです。HeteroFL（部分モデル共有）とFedGen（Data-Free Knowledge Distillation）を組み合わせることで、計算資源の異なるデバイス間での効率的な知識共有を実現します。

## 特徴

- **ハイブリッド集約戦略**:
  - **同族間 (Intra-Family)**: HeteroFL方式による、同じアーキテクチャファミリー内での部分的重み共有（例: ResNet50 ↔ ResNet18）。
  - **異種間 (Inter-Family)**: FedGen方式による、Generatorを用いたData-Free Knowledge Distillation（例: CNN ↔ ViT）。
- **動的モデル割当**: クライアントの計算資源に応じて適切なモデルを割り当てます（Phase 1では固定割当）。
- **対応モデル**:
  - **CNN Family**: ResNet (18, 50), MobileNetV3 (Large, Small)
  - **ViT Family**: ViT (Tiny, Small), DeiT (Tiny, Small)

## 動作環境

- Python 3.10+
- PyTorch >= 2.0.0
- Flower (flwr) >= 1.7.0

## セットアップ

1. **リポジトリのクローン**:
   ```bash
   git clone https://github.com/Richiesss/AFAD.git
   cd AFAD
   ```

2. **仮想環境の作成と有効化**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **依存パッケージのインストール**:
   ```bash
   pip install -r requirements.txt
   ```

## 使用方法

### 設定

`config/afad_config.yaml` で実験パラメータを設定できます。

```yaml
experiment:
  num_rounds: 50
  
server:
  min_clients: 5
  
data:
  dataset: "mnist"
  batch_size: 64
```

### 実験の実行

#### Flower Simulation (推奨)
単一マシン上で複数のクライアントをシミュレーションします。

```bash
python scripts/run_experiment.py
```

#### 手動統合テスト
基本的な学習ループの動作確認を行います（Flower Simulationが動作しない環境向け）。

```bash
python scripts/debug_integration.py
```

## ディレクトリ構造

```
afad/
├── config/                 # 設定ファイル
│   └── afad_config.yaml
├── scripts/                # 実行スクリプト
│   ├── run_experiment.py   # Flower Simulation実行
│   └── debug_integration.py # 統合テスト用
├── src/                    # ソースコード
│   ├── client/             # クライアント実装 (AFADClient)
│   ├── data/               # データローダー (MNIST)
│   ├── models/             # モデル定義 (ResNet, ViT, Registry)
│   ├── routing/            # ファミリー判定 (FamilyRouter)
│   ├── server/             # サーバー実装 (AFADStrategy, Generator)
│   └── utils/              # ユーティリティ (Logger, Config)
├── tests/                  # 単体テスト
└── requirements.txt        # 依存関係
```

## 開発者

- **作成者**: 島野 凌
- **所属**: 大阪工業大学 大学院 情報科学研究科
