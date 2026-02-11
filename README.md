# AFAD (Adaptive Federated Architecture Distribution)

異種モデルが混在する連合学習（Federated Learning）環境で、効率的に知識を共有するハイブリッドフレームワーク。
HeteroFL（同族間の部分重み共有）と FedGen（異種間の Data-Free Knowledge Distillation）を組み合わせ、計算資源が異なるデバイス間でも高精度な協調学習を実現する。

## 主な特徴

- **ハイブリッド集約戦略**
  - **同族間 (Intra-Family)**: HeteroFL による部分的重み共有（例: ResNet50 ↔ ResNet18）
  - **異種間 (Inter-Family)**: FedGen による Generator ベースの Data-Free KD（例: CNN ↔ ViT）
- **サーバーサイド知識蒸留**
  - EMA ブレンディング（β=0.1）で実データ訓練の重みを保護しつつ知識を転写
  - 品質ゲートにより低品質な合成データでの蒸留を自動スキップ
- **学習率スケジューリング**
  - Cosine Annealing で学習率を自動減衰（lr_max → lr_max×0.01）
- **5 種類の異種モデルをサポート**
  - CNN Family: ResNet50, ResNet18, MobileNetV3-Large
  - ViT Family: ViT-Tiny, DeiT-Small
- **Flower Simulation** による単一マシン上の FL シミュレーション（Ray バックエンド）
- **GPU / CPU 両対応** — CUDA 検出時は自動で GPU を使用

## Phase 1 実験結果

MNIST・IID 分布・5 クライアント・40 ラウンド・Cosine LR での 3 方式比較:

| 方式 | Best Accuracy | Final Accuracy | Total Time |
|------|:---:|:---:|:---:|
| **HeteroFL Only** | **95.58%** | **95.56%** | 1,951s |
| FedGen Only | 95.37% | 95.31% | 2,161s |
| AFAD Hybrid | 95.52% | 95.48% | 2,210s |

精度推移:

```
         HeteroFL Only    FedGen Only    AFAD Hybrid
Round  1:    49.41%          50.01%        47.72%
Round  5:    89.69%          89.20%        89.87%
Round 10:    92.96%          93.31%        92.52%
Round 20:    94.73%          94.77%        94.62%
Round 30:    95.43%          95.00%        95.48%
Round 40:    95.56%          95.31%        95.48%
```

> **Note**: 5 つの固有アーキテクチャ構成では、各クライアントが独自の signature グループを形成するため、HeteroFL 集約は単純コピーに退化する。FedGen による知識蒸留が異種モデル間の唯一の知識共有メカニズムとして機能している。3 方式とも 95% 以上の精度に収束し、安定した学習曲線を示す。

## Phase 2 実験結果

OrganAMNIST（11 クラス臓器分類）・Non-IID 分布（Dirichlet α=0.5）・10 クライアント（5 アーキテクチャ × 2）・40 ラウンド・Cosine LR:

| 方式 | Best Accuracy | Final Accuracy | Final Loss | Total Time |
|------|:---:|:---:|:---:|:---:|
| HeteroFL Only | 70.88% | 70.73% | 1.4949 | 1,514s |
| FedGen Only | 70.85% | 70.75% | 1.4407 | 1,683s |
| **AFAD Hybrid** | **71.55%** | **71.37%** | **1.4320** | 1,666s |

精度推移:

```
         HeteroFL Only    FedGen Only    AFAD Hybrid
Round  1:    13.85%          15.14%        14.42%
Round  5:    46.20%          45.52%        47.04%
Round 10:    61.93%          60.55%        62.41%
Round 20:    69.11%          68.18%        69.56%
Round 30:    70.73%          70.58%        71.34%
Round 40:    70.73%          70.75%        71.37%
```

> **Phase 2 の主な知見**:
> - **AFAD Hybrid が一貫して最高精度を達成** — Best/Final ともに HeteroFL Only を約 +0.6pt 上回る
> - Non-IID 環境で HeteroFL + FedGen の相補的効果が顕在化（Phase 1 の IID 環境では3方式ほぼ同等だった）
> - FedGen の知識蒸留はロスの低減に特に効果的（HeteroFL: 1.495 → AFAD: 1.432）
> - 10 クライアント（各アーキテクチャ 2 台）により HeteroFL のグループ内集約が有効に機能

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
  num_rounds: 40          # 連合学習のラウンド数

server:
  min_clients: 5          # 参加クライアント数

strategy:
  fedgen:
    gen_epochs: 2          # Generator 学習エポック数/ラウンド
    teacher_iters: 25      # Teacher イテレーション数/エポック
    temperature: 4.0       # KD 温度パラメータ
    distill_lr: 0.0001     # 蒸留学習率
    distill_steps: 5       # 蒸留ステップ数/ラウンド
    distill_beta: 0.1      # EMA ブレンディング係数
    distill_every: 2       # 蒸留頻度（N ラウンドに 1 回）

training:
  local_epochs: 3         # クライアントのローカル学習エポック数
  learning_rate: 0.01     # SGD 初期学習率 (Cosine Annealing で自動減衰)
  momentum: 0.9
```

### 2. シミュレーション実行

```bash
# 単一方式の実験
uv run python scripts/run_experiment.py

# 3 方式比較実験 - Phase 1 (MNIST, IID, 5 clients)
uv run python scripts/run_comparison.py

# 3 方式比較実験 - Phase 2 (OrganAMNIST, Non-IID, 10 clients)
uv run python scripts/run_comparison.py config/afad_phase2_config.yaml
```

実行すると以下が自動で行われる:

1. MNIST データのダウンロード・5 クライアントへの分割
2. 各クライアントに異種モデルを割当 (ResNet50, MobileNetV3, ResNet18, ViT-Tiny, DeiT-Small)
3. 各ラウンドで:
   - **Local Training**: 各クライアントがローカルデータで学習
   - **HeteroFL Aggregation**: 同じ signature 内で重みを集約
   - **FedGen** (Round 4〜):
     - Phase 1: Generator 学習（アンサンブル一致を最大化）
     - Phase 2: サーバーサイド知識蒸留（EMA ブレンディング付き）
   - **Evaluation**: テストデータで全クライアントの精度を評価
4. 最終サマリ・比較テーブルを出力

### 3. ログの見方

実行中、各ラウンドのメトリクスがリアルタイムで表示される:

```
Round  1: loss=1.3706, accuracy=0.4964, clients=5, time=56.4s
Round  2: loss=0.7811, accuracy=0.7382, clients=5, time=52.2s
...
Generator quality check passed: ensemble_acc=100.00%
Distilled 5 models, avg_loss=0.0479
...
Round 40: loss=0.1851, accuracy=0.9531, clients=5, time=55.3s
```

## アーキテクチャ

```
┌──────────────────────────────────────────────────────────┐
│                    Flower Server                          │
│                                                           │
│  AFADStrategy                                             │
│  ├── configure_fit()    クライアントごとのモデル配信       │
│  ├── aggregate_fit()    HeteroFL + FedGen 集約            │
│  │   ├── HeteroFLAggregator   同族間: 重み平均            │
│  │   └── FedGenDistiller      異種間: サーバーサイド KD   │
│  │       ├── Phase 1: Generator 学習                      │
│  │       │   L = α·L_teacher + η·L_diversity              │
│  │       ├── Phase 2: 品質ゲート (ensemble_acc ≥ 40%)     │
│  │       └── Phase 3: EMA 付き KD 蒸留                    │
│  │           new = (1-β)·original + β·distilled           │
│  ├── configure_evaluate()  各クライアントに自身のモデル送信│
│  └── aggregate_evaluate()  加重平均で精度集約              │
│                                                           │
│  SyntheticGenerator (MNIST 正規化済み合成画像生成)         │
│  MetricsCollector   (ラウンドごとの精度・損失・時間記録)   │
└──────────────────────┬────────────────────────────────────┘
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
│   ├── afad_config.yaml          # Phase 1 実験設定 (MNIST)
│   └── afad_phase2_config.yaml   # Phase 2 実験設定 (OrganAMNIST)
├── scripts/
│   ├── run_experiment.py         # 単一実験スクリプト
│   ├── run_comparison.py         # 3 方式比較スクリプト
│   └── debug_integration.py      # 手動統合テスト
├── src/
│   ├── client/
│   │   └── afad_client.py        # FL クライアント (NumPyClient)
│   ├── data/
│   │   ├── dataset_config.py     # データセット設定レジストリ
│   │   ├── mnist_loader.py       # MNIST データローダー (Phase 1)
│   │   └── medmnist_loader.py    # OrganAMNIST データローダー + Dirichlet 分割 (Phase 2)
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
│   │   │   └── synthetic_generator.py  # 合成画像 Generator
│   │   └── strategy/
│   │       ├── afad_strategy.py        # AFAD 統合戦略
│   │       ├── heterofl_aggregator.py  # HeteroFL 同族間集約
│   │       └── fedgen_distiller.py     # FedGen サーバーサイド蒸留
│   └── utils/
│       ├── config_loader.py      # YAML 設定読み込み
│       ├── logger.py             # ロガー
│       └── metrics.py            # MetricsCollector
├── tests/                        # テスト (77 件)
│   ├── test_integration.py       # E2E 統合テスト (5/11 クラス)
│   ├── test_fedgen_distiller.py  # FedGen 蒸留テスト (EMA 含む)
│   ├── test_heterofl_aggregator.py  # HeteroFL 集約テスト
│   ├── test_generator.py         # Generator テスト
│   ├── test_medmnist_loader.py   # Dirichlet 分割テスト
│   ├── test_dataset_config.py    # データセット設定テスト
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

### サーバーサイド知識蒸留

AFAD は FedGen (Zhu et al., ICML 2021) をサーバーサイドに適応している。
クライアント側での蒸留は合成データの品質問題により精度が不安定になるため、蒸留をサーバー側で行い、以下の安定化機構を導入した:

1. **Generator Training**: 合成画像をモデルアンサンブルに入力し、一貫した予測を生成するよう Generator を学習
2. **品質ゲート**: アンサンブルが合成画像を 40% 以上正しく分類できない場合、蒸留をスキップ
3. **Knowledge Distillation**: アンサンブルの soft logits を教師として、KL ダイバージェンスで各モデルに知識を転写

```
KD_loss = KL(softmax(student/T) ‖ softmax(teacher/T))
```

4. **EMA ブレンディング**: 蒸留後の重みを元の重みと混合し、実データ訓練の成果を保護

```
new_weights = (1 - β) × original_weights + β × distilled_weights  (β=0.1)
```

5. **周期的蒸留**: 毎ラウンドではなく 2 ラウンドに 1 回蒸留を実行し、累積的な劣化を防止

### Cosine LR Scheduling

学習率を Cosine Annealing で自動調整:

```
lr(t) = lr_min + 0.5 × (lr_max - lr_min) × (1 + cos(π × t / T))
```

初期学習率 0.01 から最終的に 0.0001 まで滑らかに減衰し、終盤の精度を安定させる。

### FedGen Warmup

FedGen 蒸留は Round 4 から開始（3 ラウンドの warmup）。
初期ラウンドでローカル学習を十分に行い、モデルが意味のある特徴を獲得してから蒸留を開始することで、学習の不安定化を防ぐ。

### 参考文献

- Diao, E. et al. "HeteroFL: Computation and Communication Efficient Federated Learning for Heterogeneous Clients" (ICLR 2021)
- Zhu, Z. et al. "Data-Free Knowledge Distillation for Heterogeneous Federated Learning" (ICML 2021)

## 開発者

- **作成者**: 島野 凌
- **所属**: 大阪工業大学 大学院 情報科学研究科
