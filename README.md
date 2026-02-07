# AFAD (Adaptive Federated Architecture Distribution)

異種モデルが混在する連合学習（Federated Learning）環境で、効率的に知識を共有するハイブリッドフレームワーク。
HeteroFL（同族間の部分重み共有）と FedGen（異種間の Data-Free Knowledge Distillation）を組み合わせ、計算資源が異なるデバイス間でも高精度な協調学習を実現する。

## 主な特徴

- **ハイブリッド集約戦略**
  - **同族間 (Intra-Family)**: HeteroFL による部分的重み共有（例: ResNet50 ↔ ResNet18）
  - **異種間 (Inter-Family)**: FedGen による Generator ベースの Data-Free KD（例: CNN ↔ ViT）
- **5 種類の異種モデルをサポート**
  - CNN Family: ResNet50, ResNet18, MobileNetV3-Large
  - ViT Family: ViT-Tiny, DeiT-Small
- **Flower Simulation** による単一マシン上の FL シミュレーション（Ray バックエンド）
- **GPU / CPU 両対応** — CUDA 検出時は自動で GPU を使用

## Phase 1 実験結果

MNIST・IID 分布・5 クライアント・20 ラウンドでの結果:

| 指標 | 値 |
|------|-----|
| 最終精度 | **96.69%** |
| 最終損失 | 0.158 |
| 95% 到達ラウンド | Round 11 |
| 総実行時間 | 約 10.8 分 (RTX 4090) |

精度推移:

```
Round  1: 64.4%  ─  Round  5: 85.3%  ─  Round 10: 94.7%
Round 11: 95.9%  ─  Round 15: 95.9%  ─  Round 20: 96.7%
```

## 動作環境

- Python 3.10+
- PyTorch >= 2.0
- Flower (flwr) >= 1.7
- CUDA 対応 GPU（推奨。CPU のみでも動作可能）

## セットアップ

```bash
git clone https://github.com/Richiesss/AFAD.git
cd AFAD
```

パッケージ管理には **uv** を使用する:

```bash
# uv のインストール（未導入の場合）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 依存パッケージのインストール
uv sync
```

## シミュレーションの実行

### 1. 設定の確認・変更

`config/afad_config.yaml` で実験パラメータを調整できる:

```yaml
experiment:
  num_rounds: 20          # 連合学習のラウンド数

server:
  min_clients: 5          # 参加クライアント数
  min_fit_clients: 5      # 学習に参加する最小クライアント数

strategy:
  fedgen:
    temperature: 4.0      # KD 温度パラメータ
    gen_steps: 10          # Generator 学習ステップ数/ラウンド
    distill_steps: 3       # 蒸留ステップ数/ラウンド
    distill_lr: 0.0001     # 蒸留学習率

training:
  local_epochs: 2         # クライアントのローカル学習エポック数
  learning_rate: 0.01     # SGD 学習率
  momentum: 0.9
```

### 2. シミュレーション実行

```bash
uv run python scripts/run_experiment.py
```

実行すると以下が自動で行われる:

1. MNIST データのダウンロード・5 クライアントへの分割
2. 各クライアントに異種モデルを割当 (ResNet50, MobileNetV3, ResNet18, ViT-Tiny, DeiT-Small)
3. 各ラウンドで:
   - **Local Training**: 各クライアントがローカルデータで学習
   - **HeteroFL Aggregation**: 同じアーキテクチャ内で重みを集約
   - **FedGen Distillation** (Round 4〜): Generator 学習 → 全モデルへの知識蒸留
   - **Evaluation**: テストデータで全クライアントの精度を評価
4. 最終サマリを出力

### 3. ログの見方

実行中、各ラウンドのメトリクスがリアルタイムで表示される:

```
Round  1: loss=0.9552, accuracy=0.6438, clients=5, time=33.0s
Round  2: loss=0.3957, accuracy=0.8685, clients=5, time=30.5s
...
Experiment Summary:
  best_accuracy: 0.9669
  final_accuracy: 0.9669
  final_loss: 0.1583
  total_wall_time: 645.4307
```

## アーキテクチャ

```
┌─────────────────────────────────────────────────────────┐
│                    Flower Server                         │
│                                                          │
│  AFADStrategy                                            │
│  ├── configure_fit()    クライアントごとのモデル配信      │
│  ├── aggregate_fit()    HeteroFL + FedGen 集約           │
│  │   ├── HeteroFLAggregator   同族間: 重み平均           │
│  │   └── FedGenDistiller      異種間: KD 蒸留            │
│  │       ├── Phase 1: Generator 学習                     │
│  │       └── Phase 2: Ensemble → 各モデル KD             │
│  ├── configure_evaluate()  各クライアントに自身のモデル送信│
│  └── aggregate_evaluate()  加重平均で精度集約             │
│                                                          │
│  SyntheticGenerator (MNIST 正規化済み合成画像生成)        │
│  MetricsCollector   (ラウンドごとの精度・損失・時間記録)  │
└──────────────────────┬──────────────────────────────────┘
                       │ Flower Protocol
    ┌──────────────────┼──────────────────┐
    │                  │                  │
┌───┴───┐  ┌──────┐  ┌┴──────┐  ┌──────┐  ┌───────┐
│Client0│  │Client1│  │Client2│  │Client3│  │Client4│
│ResNet50│ │MNetV3 │  │ResNet18│ │ViT-T │  │DeiT-S │
│  CNN  │  │  CNN  │  │  CNN  │  │  ViT │  │  ViT  │
└───────┘  └──────┘  └───────┘  └──────┘  └───────┘
```

## ディレクトリ構造

```
AFAD/
├── config/
│   └── afad_config.yaml          # 実験設定
├── scripts/
│   ├── run_experiment.py         # Flower Simulation 実行スクリプト
│   └── debug_integration.py      # 手動統合テスト
├── src/
│   ├── client/
│   │   └── afad_client.py        # FL クライアント (NumPyClient)
│   ├── data/
│   │   └── mnist_loader.py       # MNIST データローダー・分割
│   ├── models/
│   │   ├── registry.py           # モデルレジストリ (ファクトリパターン)
│   │   ├── cnn/
│   │   │   ├── resnet.py         # ResNet18, ResNet50
│   │   │   └── mobilenet.py      # MobileNetV3-Large
│   │   └── vit/
│   │       ├── vit.py            # ViT-Tiny
│   │       └── deit.py           # DeiT-Small
│   ├── routing/
│   │   └── family_router.py      # モデルファミリー判定
│   ├── server/
│   │   ├── generator/
│   │   │   └── synthetic_generator.py  # 合成画像 Generator (EMA 付き)
│   │   └── strategy/
│   │       ├── afad_strategy.py        # AFAD 統合戦略
│   │       ├── heterofl_aggregator.py  # HeteroFL 同族間集約
│   │       └── fedgen_distiller.py     # FedGen 異種間蒸留
│   └── utils/
│       ├── config_loader.py      # YAML 設定読み込み
│       ├── logger.py             # ロガー
│       └── metrics.py            # MetricsCollector
├── tests/                        # テスト (35 件)
│   ├── test_integration.py       # E2E 統合テスト
│   ├── test_fedgen_distiller.py  # FedGen 蒸留テスト
│   ├── test_heterofl_aggregator.py  # HeteroFL 集約テスト
│   ├── test_generator.py         # Generator テスト
│   ├── test_metrics.py           # MetricsCollector テスト
│   └── test_router.py            # FamilyRouter テスト
├── pyproject.toml                # プロジェクト設定・依存関係
└── uv.lock                       # 依存関係ロックファイル
```

## 開発

### テスト

```bash
uv run pytest -v
```

### Lint / Format

```bash
# チェック
uv run ruff check .
uv run ruff format --check .

# 自動修正
uv run ruff check --fix . && uv run ruff format .
```

### 一括実行

```bash
uv run poe all    # lint + test
```

## 技術的な補足

### なぜ HeteroFL + FedGen のハイブリッドか

5 つの異種アーキテクチャでは、各クライアントが固有のパラメータ構造（signature）を持つため、HeteroFL 集約は単純コピーに退化する（1 グループ 1 クライアント）。
**FedGen による Data-Free KD が、異種モデル間の唯一の知識共有メカニズム** として機能する。

### FedGen の動作

1. **Generator Training**: 合成画像をモデルアンサンブルに入力し、一貫した予測を生成するよう Generator を学習
2. **Knowledge Distillation**: アンサンブルの soft logits を教師として、KL ダイバージェンス (温度スケーリング付き) で各モデルに知識を転写

```
KD_loss = T² × KL(softmax(student/T) ‖ softmax(teacher/T))
```

### FedGen Warmup

FedGen 蒸留は Round 4 から開始（3 ラウンドの warmup）。
初期ラウンドでローカル学習を十分に行い、モデルが意味のある特徴を獲得してから蒸留を開始することで、学習の不安定化を防ぐ。

## 開発者

- **作成者**: 島野 凌
- **所属**: 大阪工業大学 大学院 情報科学研究科
