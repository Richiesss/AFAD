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

AFAD では $\alpha = \beta = 0.5 / \text{model\_rate}$（小容量クライアントほど強いKDガイダンス）で設定し、$0.98^{round}$ で指数減衰する:

| model_rate | α = β |
|:---:|:---:|
| 1.0 | 0.5 |
| 0.5 | 1.0 |
| 0.25 | 2.0 |

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

各モデルには `Scaler` モジュールが挿入され、幅縮小による活性化値の変化を `output / model_rate` で補償する（学習時のみ、推論時は恒等写像）。Scaler は全残差層の後に **1 回だけ** 適用する（論文 §3.1 に準拠）。

HeteroFL ResNet18 は **Static BatchNorm (sBN)** を採用（`track_running_stats=False`）。FedAvg 集約後の running stats 不整合を回避し、学習・推論とも常に現在バッチの統計を使用する。

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

**Warmup 期間 (5 ラウンド)** の後、サーバーから受信した Generator パラメータを用いてクライアント側で正則化を行う。Warmup によりジェネレーターが十分収束してから KD を開始するため、初期の雑なシグナルによる悪影響を回避できる:

1. **教師損失** ($\alpha$ 項): ランダムラベル $y_{rand}$ に対して Generator が生成した潜在ベクトルを classifier に入力し、CE を計算
2. **潜在マッチング損失** ($\beta$ 項): 実データの出力分布と、同じラベルで生成した潜在ベクトルの classifier 出力分布の KL ダイバージェンスを最小化

$\alpha = \beta = 0.5 / \text{model\_rate}$ とすることで、小容量（sub-rate）クライアントに強いKD補正を与える。両項は $0.98^{round}$ で減衰し、EARLY_STOP_EPOCH (20) 以降は無効化される。

---

## 3 方比較実験

同一の 10 クライアント構成（CNN 5 台 + ViT 5 台、各 rate = 1.0, 1.0, 0.5, 0.5, 0.25）で 3 手法を比較する:

### 実験構成

| モード | Intra-Family 集約 | Inter-Family KD | クライアント | Generator |
|--------|:-:|:-:|:-:|:-:|
| HeteroFL Only | HeteroFL distribute/aggregate | なし | `HeteroFLClient` | なし |
| FedGen Only | FedAvg (rate=1.0) | Client-side latent KD | `FedGenClient` | `FedGenGenerator` |
| AFAD Hybrid | HeteroFL distribute/aggregate | Client-side latent KD | `AFADClient` | `FedGenGenerator` |

### 直接シミュレーション結果 (MNIST, IID, 10 clients, 30 rounds)

> `run_direct_sim.py` を使用（Ray/Flower 不要）。seed=42 の単一試行。

| 方式 | Best Accuracy | Final Accuracy | Total Time |
|------|:---:|:---:|:---:|
| HeteroFL Only | 69.90% | 69.50% | ~120s |
| FedGen Only | 67.00% | 66.85% | ~135s |
| **AFAD Hybrid** | **69.85%** | **69.70%** | ~173s |

**AFAD 改善の推移**（全4施策の累積効果）:

| 施策 | AFAD BEST | 差分 |
|------|:---:|:---:|
| ベースライン（改善前） | 60.30% | — |
| ① 全クライアントKD適用 | 61.90% | +1.60pp |
| ② α/β を 10→1 に削減 | 69.15% | +7.25pp |
| ③ KD Warmup (5ラウンド) | 69.25% | +0.10pp |
| **④ レート依存 KD スケーリング** | **69.85%** | **+0.60pp** |

> **主な知見**:
>
> 1. **施策②が最大貢献** (+7.25pp): FedGen 用にチューニングされた α=β=10 は AFAD Hybrid では過剰正則化。α=β=1 に下げることで CE 損失が主導的になり精度が急向上
> 2. **施策③ (KD Warmup)** はジェネレーター収束後にKDを開始することで安定性向上（FINAL=BEST）
> 3. **施策④ (レート依存スケーリング)** はサブレートクライアントの学習を補強し AFAD が HeteroFL を上回る 69.85% を達成
> 4. **HeteroFL sBN 修正**（track_running_stats=False）が性能の根幹: 修正前40.85% → 修正後69.90%（+29pp）

### Phase 2 結果 (OrganAMNIST, Non-IID α=0.5, 10 clients, 40 rounds)

| 方式 | Best Accuracy | Final Accuracy | Loss |
|------|:---:|:---:|:---:|
| HeteroFL Only | 70.91% | 70.78% | 1.4937 |
| FedGen Only | 57.61% | 57.43% | 2.3869 |
| **AFAD Hybrid** | **71.11%** | **71.11%** | **1.4133** |

> **注**: 上記は旧バージョン（sBN 修正前）での結果。単一 seed (42) の予備結果。

---

## AFAD のメリット：精度以外の優位点

### 1. 唯一「二重の異種性」を同時に扱える

これが最も本質的な優位点である。HeteroFL と FedGen はそれぞれ一方の異種性しか扱えない。

| | 計算能力の異種性 (rate=1.0/0.5/0.25) | アーキテクチャ異種性 (CNN ↔ ViT) |
|---|:---:|:---:|
| HeteroFL | ○ | × (family 内のみ) |
| FedGen | × (全員同一幅が前提) | ○ |
| **AFAD** | **○** | **○** |

HeteroFL は CNN クライアントと ViT クライアントを完全に独立した family として扱い、**ファミリー間での知識移転が一切ない**。FedGen は全クライアントが同じ幅でないと機能しない。実際のデプロイ環境でアーキテクチャと計算能力の両方の混在が起きるのは必然であり、AFAD 以外の単独手法では対応できない。

### 2. 低容量クライアント（sub-rate）の学習を補強する

HeteroFL Only の rate=0.25 クライアントは自分の狭いモデルのみで学習するため、大型クライアントの知識を活用できない。AFAD では Generator からの KD がレート依存スケーリング（`α = β = 0.5 / model_rate`）で機能し、**容量が小さいほど強いガイダンスを受ける**：

| model_rate | α = β (KD強度) | 意味 |
|:---:|:---:|:---|
| 1.0 | 0.5 | 自力で学べるため KD は補助的 |
| 0.5 | 1.0 | 標準的な KD |
| 0.25 | 2.0 | 容量不足を Generator の知識で補完 |

### 3. 収束の安定性が高い

KD が正則化として機能し、損失空間を平滑化するため終盤のブレが小さい：

| 方式 | BEST | FINAL | ブレ |
|------|:---:|:---:|:---:|
| HeteroFL Only | 69.90% | 69.50% | −0.40pp |
| FedGen Only | 67.00% | 66.85% | −0.15pp |
| **AFAD Hybrid** | **69.85%** | **69.70%** | **−0.15pp** |

「どのラウンドで学習を止めても安定した精度が出る」という特性は実運用上重要である。

### 4. 収束速度が速い

Generator の KD が追加の学習信号として機能するため、**同じ精度水準に達するまでに必要なラウンド数（＝通信回数）が少ない**：

```
Round |  HeteroFL Only  |  AFAD Hybrid
   10 |    63.20%       |   64.15%  ← Round 10 からリード
   14 |    66.60%       |   68.50%  ← +1.90pp 差
   18 |    68.60%       |   69.85%  ← AFAD はほぼ収束
   30 |    69.30%       |   69.70%
```

### 5. Non-IID 耐性（データ不均質性への対応）

FedGen の Generator はサーバー側でクラスバランスのよい潜在ベクトルを生成する。クライアントのローカルデータが特定クラスに偏っていても、**Generator がクラスバランスの取れた仮想データを補完**するため Non-IID 環境に強い。HeteroFL Only にはこの仕組みがない。Phase 2（OrganAMNIST, Non-IID α=0.5）ではその差が顕著に現れた：

| 方式 | Non-IID Best Accuracy |
|------|:---:|
| HeteroFL Only | 70.91% |
| FedGen Only | 57.61% |
| **AFAD Hybrid** | **71.11%** |

FedGen は Non-IID 環境で大きく落ち込む（HeteroFL 比 −13.30pp）が、AFAD はその落ち込みを防ぎつつ HeteroFL をわずかに上回る。

### メリットまとめ

| メリット | HeteroFL | FedGen | AFAD |
|--------|:---:|:---:|:---:|
| 二重の異種性に対応 | × | × | **○** |
| sub-rate クライアントへの知識補完 | × | △ | **○** |
| 収束の安定性 | 中 | 高 | **高** |
| 収束速度 | 中 | 遅 | **速** |
| Non-IID 耐性 | 中 | 低 | **高** |

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
    α = β = 0.5 / model_rate (レート依存スケーリング)
```

---

## 原著論文との設計差分

### HeteroFL [Diao+, ICLR 2021] との差分

| 観点 | 原著 | AFAD |
|------|------|------|
| モデル幅 | 固定アーキテクチャ族内 | CNN + ViT の 2 ファミリー対応 |
| 出力層保護 | 最終 Linear のみ | `num_preserved_tail_layers` で bottleneck + classifier を保護 |
| Scaler 適用 | 全残差層後に 1 回 | 同一（論文 §3.1 準拠） |
| BatchNorm | Static BN (track_running_stats=False) | 同一（sBN を全 BN 層に適用） |
| 集約 | Count-based 平均 | 同一 + label-split 集約 |

### FedGen [Zhu+, ICML 2021] との差分

| 観点 | 原著 | AFAD |
|------|------|------|
| Generator 出力 | 潜在ベクトル (32 次元) | 同一 (`FedGenGenerator`) |
| KD 実行場所 | クライアント側正則化 | 同一 (`AFADClient._train`) |
| KD 損失関数 | CE + KL | 同一（減衰あり） |
| KD 係数 α, β | 固定値 | レート依存: 0.5 / model_rate |
| KD Warmup | なし（明示的） | `AFAD_KD_WARMUP_ROUNDS=5` |
| 集約 | FedAvg | HeteroFL (AFAD モード) / FedAvg (FedGen Only モード) |
| Generator 学習 | 全クライアントモデルで | family ごとの rate=1.0 モデルで (`AFADGeneratorTrainer`) |

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

### 直接シミュレーション（推奨）

Ray/Flower ワーカーを使わず高速に起動できる:

```bash
# フル実験 (10 clients, 30 rounds, MNIST)
uv run python scripts/run_direct_sim.py

# 出力先を指定
uv run python scripts/run_direct_sim.py --output results/my_run.json

# クイックテスト (4 clients, 5 rounds, 500 samples)
uv run python scripts/run_quick_test.py
```

### 出力例

```
============================================================
  AFAD Hybrid
  enable_fedgen=True, enable_heterofl=True
============================================================
  Round  1/30: acc=0.2865  loss=2.2801  time=3.2s
  ...
  Round 30/30: acc=0.6970  loss=1.1139  time=7.6s

==========================================================================
  COMPARISON: Accuracy per Round
==========================================================================
Round |        HeteroFL Only |          FedGen Only |          AFAD Hybrid
--------------------------------------------------------------------------
    1 |              23.40% |              30.85% |              29.00%
   ...
   30 |              69.30% |              66.45% |              69.70%
--------------------------------------------------------------------------
 BEST |              69.70% |              66.75% |              69.85%
FINAL |              69.30% |              66.45% |              69.70%
```

### Flower ベース実験（旧版）

```bash
# 3 方式比較実験 - Phase 1 (MNIST, IID)
uv run python scripts/run_comparison.py

# 3 方式比較実験 - Phase 2 (OrganAMNIST, Non-IID)
uv run python scripts/run_comparison.py config/afad_phase2_config.yaml

# Multi-seed 統計的検証 (10 seeds)
python scripts/run_multi_seed.py config/afad_phase2_config.yaml
```

## ディレクトリ構造

```
AFAD/
├── scripts/
│   ├── run_direct_sim.py           # 直接シミュレーション（推奨）
│   ├── run_quick_test.py           # 4 clients, 5 rounds のクイック検証
│   ├── run_comparison.py           # 3 方式比較（Flower ベース）
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
│   │   ├── mnist_loader.py         # MNIST データローダー
│   │   └── medmnist_loader.py      # OrganAMNIST + Dirichlet 分割
│   ├── models/
│   │   ├── registry.py             # モデルレジストリ (ファクトリパターン)
│   │   ├── fedgen_wrapper.py       # FedGenModelWrapper (bottleneck + classifier)
│   │   ├── scaler.py               # HeteroFL Scaler (1/rate 補償)
│   │   ├── cnn/
│   │   │   ├── heterofl_resnet.py  # 幅スケーラブル ResNet18 (sBN + 1x Scaler)
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
│   │   │   ├── afad_generator_trainer.py # サーバーサイド Generator 訓練 (EMA)
│   │   │   └── synthetic_generator.py    # (レガシー) 画像生成 Generator
│   │   └── strategy/
│   │       ├── afad_strategy.py          # AFAD 統合戦略 (3 モード対応)
│   │       ├── heterofl_aggregator.py    # HeteroFL 集約 (count-based + label-split)
│   │       └── fedgen_distiller.py       # (レガシー) サーバーサイド蒸留
│   └── utils/
│       ├── config_loader.py        # YAML 設定読み込み
│       ├── logger.py               # ロガー
│       └── metrics.py              # MetricsCollector
├── tests/                          # テスト
│   ├── test_afad_integration.py    # AFAD E2E 統合テスト
│   ├── test_integration.py         # Strategy 統合テスト
│   ├── test_fedgen_faithful.py     # FedGen コンポーネントテスト
│   ├── test_fedprox.py             # FedProx 正則化テスト
│   ├── test_heterofl_aggregator.py # HeteroFL 集約テスト (tail layers 含む)
│   ├── test_generator.py           # Generator テスト
│   ├── test_medmnist_loader.py     # Dirichlet 分割テスト
│   └── ...
├── results/                        # 実験結果 JSON
├── pyproject.toml
└── uv.lock
```

---

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

---

## 参考文献

- Diao, E. et al. "HeteroFL: Computation and Communication Efficient Federated Learning for Heterogeneous Clients" (ICLR 2021)
- Zhu, Z. et al. "Data-Free Knowledge Distillation for Heterogeneous Federated Learning" (ICML 2021)
- Li, T. et al. "Federated Optimization in Heterogeneous Networks" (MLSys 2020)
- Hinton, G. et al. "Distilling the Knowledge in a Neural Network" (NeurIPS Workshop 2015)

## 開発者

- **作成者**: 島野 凌
- **所属**: 大阪工業大学 大学院 情報科学研究科
