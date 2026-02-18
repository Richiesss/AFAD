# AFAD (Adaptive Federated Architecture Distribution)

## 概要

連合学習 (Federated Learning; FL) では、参加デバイスの計算資源やモデルアーキテクチャが不均一であることが一般的である。AFAD は、この**二重の異種性**（計算能力 + アーキテクチャ）に対処するハイブリッドフレームワークである。

AFAD は以下の 2 つの独立した研究を統合する:

| 手法 | 対処する問題 | メカニズム |
|------|------------|-----------|
| **HeteroFL** [Diao+, ICLR 2021] | 計算能力の異種性 | 同一アーキテクチャの幅スケーリング（model_rate）による部分重み共有 |
| **FedGen** [Zhu+, ICML 2021] | アーキテクチャの異種性 | 潜在空間での Data-Free Knowledge Distillation |

| 手法 | 計算の異種性 | アーキテクチャの異種性 |
|------|:---:|:---:|
| HeteroFL 単体 | ○ (幅スケーリング) | × (同一アーキテクチャのみ) |
| FedGen 単体 | × (全クライアント同一サイズ) | ○ (潜在空間で知識共有) |
| **AFAD** | **○** | **○** |

---

## 設計思想

### 共有潜在空間

AFAD の核心は **共有潜在空間** (shared latent space, 32 次元) の導入である。`FedGenModelWrapper` が全モデルに `bottleneck(feature_dim → 32) + classifier(32 → num_classes)` を追加し、backbone のアーキテクチャや幅に関係なく同一の潜在空間を共有する。

```
CNN family (HeteroFL ResNet18)          ViT family (HeteroFL ViT-Small)
┌──────────────────────┐                ┌──────────────────────┐
│ backbone (幅可変)     │                │ backbone (幅可変)     │
│ rate=1.0: 512ch      │                │ rate=1.0: 384dim     │
│ rate=0.5: 256ch      │                │ rate=0.5: 192dim     │
│ rate=0.25: 128ch     │                │ rate=0.25: 96dim     │
├──────────────────────┤                ├──────────────────────┤
│ bottleneck           │                │ bottleneck           │
│ feature_dim → 32     │                │ feature_dim → 32     │
├──────────────────────┤                ├──────────────────────┤
│ classifier           │                │ classifier           │
│ 32 → num_classes     │                │ 32 → num_classes     │
└──────────────────────┘                └──────────────────────┘
         ↑                                        ↑
         └──────── 共有潜在空間 (32次元) ──────────┘
                           ↑
                    ┌──────────────┐
                    │ FedGen       │
                    │ Generator    │
                    │ → 32次元出力  │
                    └──────────────┘
```

**潜在空間を使う理由**:

1. **32 次元ベクトルの生成は容易**: 小さな MLP で十分（画像生成よりはるかに効率的）
2. **アーキテクチャ非依存**: `forward_from_latent(z)` は classifier のみ使用し、backbone をバイパス
3. **低コスト**: classifier は `Linear(32 → 10)` の 1 層のみ

### クライアント側蒸留

原著 FedGen に準拠し、**クライアント側蒸留** を採用する。クライアントは実データで学習しつつ、生成器の出力で正則化する:

$$\mathcal{L} = \underbrace{\text{CE}(f(x), y)}_{\text{予測損失}} + \alpha \cdot \underbrace{\text{CE}(\text{classifier}(G(y_{rand})), y_{rand})}_{\text{教師損失}} + \beta \cdot \underbrace{\text{KL}(f(x) \| \text{classifier}(G(y_{real})))}_{\text{潜在マッチング損失}}$$

ここで $\alpha = \beta = 10.0$、$0.98^{round}$ で指数減衰する。

### HeteroFL × FedGenModelWrapper の統合

HeteroFL は `model_rate` に応じてパラメータの幅をスケールするが、`FedGenModelWrapper` が追加する bottleneck の出力次元 (latent_dim=32) と classifier は**固定**でなければならない。

`num_preserved_tail_layers` パラメータにより、末尾から N 個の 2D パラメータを保護する:

- **vanilla HeteroFL**: `num_preserved_tail_layers=1`（classifier のみ保護）
- **AFAD**: `num_preserved_tail_layers=2`（bottleneck + classifier を保護）

```
rate=0.5 (num_preserved_tail_layers=2)
backbone最終層: [256, 256, 3, 3]  ← 正しくスケール
bottleneck:     [32, 256]         ← 出力32保護、入力はスケール済み ✓
classifier:     [10, 32]          ← 完全保護 ✓
```

---

## 提案手法の詳細

### 1. 問題設定

$K$ 台のクライアントが 2 つのモデルファミリー (CNN, ViT) に分かれ、各ファミリー内で異なる幅 (model_rate) のサブモデルを持つ:

| Family | モデル | rate=1.0 | rate=0.5 | rate=0.25 |
|--------|--------|:---:|:---:|:---:|
| CNN | HeteroFL ResNet18 | 512ch, ~11.2M | 256ch, ~2.8M | 128ch, ~0.7M |
| ViT | HeteroFL ViT-Small | 384dim, ~21.3M | 192dim, ~5.4M | 96dim, ~1.4M |

各モデルには `Scaler` モジュールが挿入され、幅縮小による活性化値の変化を `output / model_rate` で補償する（学習時のみ、推論時は恒等写像）。

デフォルト構成は 10 クライアント:

| Client | Family | Model | Rate |
|--------|--------|-------|------|
| 0, 1 | CNN | heterofl_resnet18 | 1.0 |
| 2, 3 | CNN | heterofl_resnet18 | 0.5 |
| 4 | CNN | heterofl_resnet18 | 0.25 |
| 5, 6 | ViT | heterofl_vit_small | 1.0 |
| 7, 8 | ViT | heterofl_vit_small | 0.5 |
| 9 | ViT | heterofl_vit_small | 0.25 |

### 2. ラウンドごとの処理フロー

```
Round r:
┌─── Server ──────────────────────────────────────────────────┐
│ 1. configure_fit: HeteroFL distribute                       │
│    - family_global_models から sub-model を抽出              │
│    - num_preserved_tail_layers=2 で bottleneck/classifier 保護│
│    - generator params を pickle → config に含める            │
│    - Cosine LR を計算して配信                                │
│                                                             │
│ 2. [Client training は並列実行]                              │
│                                                             │
│ 3. aggregate_fit:                                           │
│    ├─ family ごとにグループ化                                │
│    ├─ HeteroFL: count-based 平均 + label-split 集約         │
│    ├─ family_global_models を更新                            │
│    └─ Generator training:                                   │
│       - family_global_models → FedGenModelWrapper 再構築     │
│       - forward_from_latent で教師損失                       │
│       - diversity loss でモード崩壊防止                      │
└─────────────────────────────────────────────────────────────┘
         │ sub-model params                    ▲ trained params
         │ + generator params                  │ + label_counts
         ▼                                     │
┌─── Client (cid=2, CNN, rate=0.5) ────────────┐
│ 1. shape-aware set_parameters                 │
│ 2. generator params を受信・設定               │
│ 3. Local training (各バッチ):                  │
│    - CE(model(x), y)                  # 実データ│
│    - α × CE(classifier(G(y_rand)), y_rand)    │
│    - β × KL(model(x) ‖ classifier(G(y_real))) │
│    - FedProx: (μ/2) ‖w - w_global‖²          │
│    - α, β は 0.98^round で減衰                │
│ 4. Return: trained params + label_counts      │
└───────────────────────────────────────────────┘
```

### 3. ローカル学習と FedProx 正則化

各クライアントは SGD + Cosine LR + Gradient Clipping (max_norm=1.0) で学習する。Non-IID 環境でのクライアントドリフトを抑制するため、FedProx [Li+, MLSys 2020] の近接項をオプションで追加できる:

$$\mathcal{L}_{local} = \mathcal{L}_{CE}(f(x), y) + \frac{\mu}{2} \| w - w^{t} \|^2$$

学習率はサーバーが Cosine Annealing で管理し、全クライアントに配信する:

$$\text{lr}(t) = \text{lr}_{min} + \frac{1}{2}(\text{lr}_{max} - \text{lr}_{min})\left(1 + \cos\left(\frac{\pi \cdot t}{T}\right)\right)$$

### 4. HeteroFL による family 内集約

各クライアントの更新パラメータを **family** ごとにグループ化し、count-based の重み平均で集約する。HeteroFL の核心は、異なる幅のモデルでも **パラメータ空間の先頭部分を共有** できる点にある:

$$w^{t+1}[i,j] = \frac{1}{|S_{i,j}|} \sum_{k \in S_{i,j}} w_{k}^{t}[i,j]$$

ここで $S_{i,j}$ はパラメータ位置 $(i,j)$ を更新したクライアントの集合。FedAvg とは異なり、**サンプル数ではなく更新カウント**で平均する。

Non-IID 環境では **label-split 集約** を適用し、出力層の各行（クラス）は当該クラスのデータを持つクライアントのみが更新する。

### 5. FedGen によるサーバーサイド Generator 学習

Generator $G$ はクラスラベル $y$ を入力とし、32 次元の潜在ベクトル $z = G(y)$ を生成する。全 family モデルの `forward_from_latent(z)` が正しいクラスを予測するよう最適化される:

$$\mathcal{L}_{G} = \alpha \cdot \mathcal{L}_{teacher} + \eta \cdot \mathcal{L}_{diversity}$$

**教師損失**: 各 family モデル $f_m$ の classifier が $z$ からラベル $y$ を正しく予測するよう重み付き CE を計算:

$$\mathcal{L}_{teacher} = \sum_{m=1}^{M} \sum_{i=1}^{B} w_{y_i, m} \cdot \text{CE}(f_m.\text{forward\_from\_latent}(z_i), y_i)$$

**多様性損失**: Generator のモード崩壊を防止:

$$\mathcal{L}_{diversity} = \exp\left(-\text{mean}(d(z) \odot d(\epsilon))\right)$$

### 6. クライアント側 FedGen 正則化

Warmup 期間 (3 ラウンド) 後、サーバーから受信した Generator パラメータを用いてクライアント側で正則化を行う:

1. **教師損失** ($\alpha$ 項): ランダムラベル $y_{rand}$ に対して Generator が生成した潜在ベクトルを classifier に入力し、CE を計算
2. **潜在マッチング損失** ($\beta$ 項): 実データの出力分布と、同じラベルで生成した潜在ベクトルの classifier 出力分布の KL ダイバージェンスを最小化

両項は $0.98^{round}$ で減衰し、EARLY_STOP_EPOCH (20) 以降は無効化される。

---

## 3 方比較実験

同一の 10 クライアント構成（CNN 5 台 + ViT 5 台、各 rate = 1.0, 1.0, 0.5, 0.5, 0.25）で 3 手法を比較する:

### 実験構成

| モード | Intra-Family 集約 | Inter-Family KD | クライアント | Generator |
|--------|:-:|:-:|:-:|:-:|
| HeteroFL Only | HeteroFL distribute/aggregate | なし | `HeteroFLClient` | なし |
| FedGen Only | FedAvg (rate=1.0) | Client-side latent KD | `FedGenClient` | `FedGenGenerator` |
| AFAD Hybrid | HeteroFL distribute/aggregate | Client-side latent KD | `AFADClient` | `FedGenGenerator` |

### Phase 1 結果 (MNIST, IID, 5 clients, 40 rounds)

| 方式 | Best Accuracy | Final Accuracy | Total Time |
|------|:---:|:---:|:---:|
| **HeteroFL Only** | **95.58%** | **95.56%** | 1,951s |
| FedGen Only | 95.37% | 95.31% | 2,161s |
| AFAD Hybrid | 95.52% | 95.48% | 2,210s |

### Phase 2 結果 (OrganAMNIST, Non-IID α=0.5, 10 clients, 40 rounds)

| 方式 | Best Accuracy | Final Accuracy | Loss |
|------|:---:|:---:|:---:|
| HeteroFL Only | 70.91% | 70.78% | 1.4937 |
| FedGen Only | 57.61% | 57.43% | 2.3869 |
| **AFAD Hybrid** | **71.11%** | **71.11%** | **1.4133** |

> **Phase 2 の主な知見**:
>
> 1. **HeteroFL 集約が支配的要因**: 集約ありの 2 方式 (70.9–71.1%) vs 集約なし (57.6%) で **13pt** の差
> 2. **AFAD Hybrid が最高精度** (71.11%)。HeteroFL Only に対して +0.20pt
> 3. **FedGen KD は正だが限定的**: AFAD − HeteroFL = +0.20pt が KD の純粋な貢献分
> 4. **FedGen 単体の限界**: KD のみでは同族間の重み共有を代替できない
>
> **注**: 上記は単一 seed (42) での予備結果。Multi-seed 統計的検証で有意性を確認する必要がある。

---

## システムアーキテクチャ

```
Server (AFADStrategy)
│
├─ configure_fit()
│   ├─ _initialize_family_models(): family ごとに rate=1.0 グローバルモデルを作成
│   ├─ HeteroFL distribute: family_global_models から sub-model を抽出
│   ├─ Cosine LR 計算 + training config を配信
│   └─ Generator params を serialize (pickle) して config に含める
│
├─ aggregate_fit()
│   ├─ Family ごとにグループ化 (client metrics の family から判定)
│   ├─ enable_heterofl=True → _aggregate_heterofl() (count-based + label-split)
│   │   enable_heterofl=False → _aggregate_fedavg() (重み付き平均)
│   └─ _train_generator_on_server()
│       ├─ family_global_models → FedGenModelWrapper を再構築
│       ├─ forward_from_latent で教師損失を計算
│       └─ Generator パラメータを更新
│
├─ configure_evaluate()
│   └─ 各クライアントに対応する family の sub-model を配信
│
└─ aggregate_evaluate()
    └─ 加重平均で全体精度を集約

Clients
├─ HeteroFLClient (HeteroFL Only baseline)
│   SGD + FedProx, shape-aware set_parameters, KD なし
├─ FedGenClient (FedGen Only baseline)
│   SGD + client-side KD (teacher + latent matching), FedAvg 前提
└─ AFADClient (AFAD Hybrid)
    SGD + FedProx + shape-aware set_parameters + client-side KD
```

---

## 原著論文との設計差分

### HeteroFL [Diao+, ICLR 2021] との差分

| 観点 | 原著 | AFAD |
|------|------|------|
| モデル幅 | 固定アーキテクチャ族内 | CNN + ViT の 2 ファミリー対応 |
| 出力層保護 | 最終 Linear のみ | `num_preserved_tail_layers` で bottleneck + classifier を保護 |
| Scaler | `1/rate` 補償 (training) | 同一 (`Scaler` モジュール) |
| 集約 | Count-based 平均 | 同一 + label-split 集約 |

### FedGen [Zhu+, ICML 2021] との差分

| 観点 | 原著 | AFAD |
|------|------|------|
| Generator 出力 | 潜在ベクトル (32 次元) | 同一 (`FedGenGenerator`) |
| KD 実行場所 | クライアント側正則化 | 同一 (`AFADClient._train`) |
| KD 損失関数 | CE + KL | 同一 (α=10, β=10, 0.98/round 減衰) |
| 集約 | FedAvg | HeteroFL (AFAD モード) / FedAvg (FedGen Only モード) |
| Generator 学習 | 全クライアントモデルで | family ごとの rate=1.0 モデルで (`AFADGeneratorTrainer`) |
| Warmup | なし (明示的) | 3 ラウンドの warmup |

---

## 動作環境

- Python 3.10+
- PyTorch >= 2.0
- Flower (flwr) >= 1.7
- CUDA 対応 GPU（推奨。CPU のみでも動作可能）

## セットアップ

```bash
git clone https://github.com/Richiesss/AFAD.git
cd AFAD

# uv のインストール（未導入の場合）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 依存パッケージのインストール
uv sync
```

## シミュレーションの実行

### 1. 設定

`config/afad_config.yaml` で実験パラメータを調整できる:

```yaml
experiment:
  num_rounds: 40

server:
  min_clients: 10

training:
  local_epochs: 3
  learning_rate: 0.01
  momentum: 0.9
```

### 2. 実行

```bash
# 3 方式比較実験 - Phase 1 (MNIST, IID)
uv run python scripts/run_comparison.py

# 3 方式比較実験 - Phase 2 (OrganAMNIST, Non-IID)
uv run python scripts/run_comparison.py config/afad_phase2_config.yaml

# Multi-seed 統計的検証 (10 seeds)
python scripts/run_multi_seed.py config/afad_phase2_config.yaml

# seed を指定して実行
python scripts/run_multi_seed.py config/afad_phase2_config.yaml --seeds 42 123 456

# 出力先を指定
python scripts/run_multi_seed.py config/afad_phase2_config.yaml --output results/my_results.json
```

> **Note**: `run_multi_seed.py` は `uv run` ではなく `python` (venv 直接) で実行すること。Ray ワーカーの runtime_env 構築問題を回避するため。結果は JSON に逐次保存され、中断後の再実行で完了済み seed を自動スキップする。

### 3. ログの見方

```
Round  1: loss=1.3706, accuracy=0.4964, clients=10, time=56.4s
Round  2: loss=0.7811, accuracy=0.7382, clients=10, time=52.2s
...
Generator trained with 2 family models
...
Round 40: loss=0.1851, accuracy=0.9531, clients=10, time=55.3s
```

## ディレクトリ構造

```
AFAD/
├── config/
│   ├── afad_config.yaml            # Phase 1 実験設定 (MNIST)
│   └── afad_phase2_config.yaml     # Phase 2 実験設定 (OrganAMNIST)
├── scripts/
│   ├── run_comparison.py           # 3 方式比較スクリプト
│   ├── run_multi_seed.py           # Multi-seed 統計的検証
│   ├── run_experiment.py           # 単一実験スクリプト
│   └── debug_integration.py        # 手動統合テスト
├── src/
│   ├── client/
│   │   ├── afad_client.py          # AFAD ハイブリッドクライアント (HeteroFL + FedGen)
│   │   ├── heterofl_client.py      # HeteroFL Only ベースラインクライアント
│   │   └── fedgen_client.py        # FedGen Only ベースラインクライアント
│   ├── data/
│   │   ├── dataset_config.py       # データセット設定レジストリ
│   │   ├── mnist_loader.py         # MNIST データローダー (Phase 1)
│   │   └── medmnist_loader.py      # OrganAMNIST + Dirichlet 分割 (Phase 2)
│   ├── models/
│   │   ├── registry.py             # モデルレジストリ (ファクトリパターン)
│   │   ├── fedgen_wrapper.py       # FedGenModelWrapper (bottleneck + classifier)
│   │   ├── scaler.py               # HeteroFL Scaler (1/rate 補償)
│   │   ├── cnn/
│   │   │   ├── heterofl_resnet.py  # 幅スケーラブル ResNet18
│   │   │   ├── resnet.py           # ResNet18, ResNet50
│   │   │   └── mobilenet.py        # MobileNetV3-Large
│   │   └── vit/
│   │       ├── heterofl_vit.py     # 幅スケーラブル ViT-Small
│   │       ├── vit.py              # ViT-Tiny, ViT-Small
│   │       └── deit.py             # DeiT-Small
│   ├── routing/
│   │   └── family_router.py        # モデルファミリー判定
│   ├── server/
│   │   ├── generator/
│   │   │   ├── fedgen_generator.py       # FedGen 潜在空間 Generator
│   │   │   ├── afad_generator_trainer.py # サーバーサイド Generator 訓練
│   │   │   └── synthetic_generator.py    # (レガシー) 画像生成 Generator
│   │   └── strategy/
│   │       ├── afad_strategy.py          # AFAD 統合戦略 (3 モード対応)
│   │       ├── heterofl_aggregator.py    # HeteroFL 集約 (count-based + label-split)
│   │       └── fedgen_distiller.py       # (レガシー) サーバーサイド蒸留
│   └── utils/
│       ├── config_loader.py        # YAML 設定読み込み
│       ├── logger.py               # ロガー
│       └── metrics.py              # MetricsCollector
├── tests/                          # テスト (132 件)
│   ├── test_afad_integration.py    # AFAD E2E 統合テスト
│   ├── test_integration.py         # Strategy 統合テスト (7 件)
│   ├── test_fedgen_faithful.py     # FedGen コンポーネントテスト
│   ├── test_fedprox.py             # FedProx 正則化テスト
│   ├── test_heterofl_aggregator.py # HeteroFL 集約テスト (tail layers 含む)
│   ├── test_fedgen_distiller.py    # FedGen 蒸留テスト (レガシー)
│   ├── test_generator.py           # Generator テスト
│   ├── test_medmnist_loader.py     # Dirichlet 分割テスト
│   ├── test_dataset_config.py      # データセット設定テスト
│   ├── test_metrics.py             # MetricsCollector テスト
│   └── test_router.py              # FamilyRouter テスト
├── pyproject.toml
└── uv.lock
```

## 開発

### テスト

```bash
uv run pytest -v              # 全テスト
uv run pytest tests/test_afad_integration.py -v  # AFAD 統合テストのみ
uv run pytest -x              # 最初の失敗で停止
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

## 参考文献

- Diao, E. et al. "HeteroFL: Computation and Communication Efficient Federated Learning for Heterogeneous Clients" (ICLR 2021)
- Zhu, Z. et al. "Data-Free Knowledge Distillation for Heterogeneous Federated Learning" (ICML 2021)
- Li, T. et al. "Federated Optimization in Heterogeneous Networks" (MLSys 2020)
- Hinton, G. et al. "Distilling the Knowledge in a Neural Network" (NeurIPS Workshop 2015)

## 開発者

- **作成者**: 島野 凌
- **所属**: 大阪工業大学 大学院 情報科学研究科
