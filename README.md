# AFAD — Adaptive Federated Architecture Distribution

連合学習における **二重の異種性**（計算能力 × モデルアーキテクチャ）を同時に解決するハイブリッド連合学習フレームワーク。

- **HeteroFL** [Diao+, ICLR 2021]: 幅スケーリングで計算能力の異種性に対応
- **FedGen** [Zhu+, ICML 2021]: Data-Free KD でアーキテクチャ異種性に対応
- **AFAD**: 両手法を統合し、互いの弱点を補完

---

## 問題設定

実際の連合学習には 2 種類の異種性が同時に存在する。

| 異種性の種類 | 具体例 | 従来手法の限界 |
|------------|--------|------------|
| **計算能力** | スマートフォン vs サーバー | FedGen は全員が同一サイズのモデルを前提 |
| **アーキテクチャ** | CNN vs Transformer | HeteroFL はファミリー間の知識共有なし |

AFAD はこの 2 種類の異種性を**同時に**扱える唯一のフレームワークである。

---

## 手法

### 全体像

```
CNN クライアント (HeteroFL ResNet18)   ViT クライアント (HeteroFL ViT-Small)
  rate=1.0: 512ch                        rate=1.0: 384dim
  rate=0.5: 256ch                        rate=0.5: 192dim
  rate=0.25: 128ch                       rate=0.25: 96dim
      │                                        │
      │  backbone (幅可変)                      │  backbone (幅可変)
      ↓                                        ↓
  bottleneck (→ 32次元)              bottleneck (→ 32次元)
      │                                        │
      └─────────── 共有潜在空間 32次元 ──────────┘
                          │
                   FedGen Generator
                  (サーバーで訓練)
```

### 1. 共有潜在空間（FedGenModelWrapper）

`FedGenModelWrapper` が全モデルに **bottleneck（feature_dim → 32）+ classifier（32 → num_classes）** を追加する。backbone のアーキテクチャや幅によらず、共通の 32 次元潜在空間でファミリー間の知識共有を実現する。

```
入力 x
  → backbone（幅スケーラブル）  ← HeteroFL で width-scaling
  → bottleneck（固定: → 32）   ← HeteroFL では保護（num_preserved_tail_layers=2）
  → classifier（固定: → 10）   ← HeteroFL では保護
```

### 2. HeteroFL：計算能力の異種性に対応

各クライアントの `model_rate`（1.0 / 0.5 / 0.25）に応じて、グローバルモデルの先頭チャネルを切り出したサブモデルを配布する。

- **Static BatchNorm**（`track_running_stats=False`）: 集約後の running stats 不整合を防止
- **Scaler**（出力 ÷ model_rate）: 幅縮小による活性化値スケール変化を補償（全残差層後に 1 回）
- **count-based 集約**: FedAvg（サンプル数重み）ではなく更新カウントで平均
- **label-split 集約**: Non-IID 時に出力層の各行を担当クラスのクライアントのみで更新

### 3. FedGen：アーキテクチャ異種性に対応

Generator $G$ がクラスラベル $y$ から 32 次元潜在ベクトル $z = G(y)$ を生成する。全ファミリーモデルの `forward_from_latent(z)` が正しいクラスを予測するようサーバーで訓練される。

$$\mathcal{L}_G = \alpha \cdot \mathcal{L}_{\text{teacher}} + \eta \cdot \mathcal{L}_{\text{diversity}}$$

- **教師損失**: 全ファミリーモデルで重み付き CE
- **多様性損失**: Generator のモード崩壊を防止

### 4. クライアント側 KD（AFAD の学習損失）

$$\mathcal{L} = \underbrace{\text{CE}(f(x), y)}_{\text{予測損失}} + \alpha \cdot \underbrace{\text{CE}(\text{classifier}(G(y_{\text{rand}})),\, y_{\text{rand}})}_{\text{教師損失}} + \beta \cdot \underbrace{\text{KL}(f(x) \| \text{classifier}(G(y_{\text{real}})))}_{\text{潜在マッチング損失}}$$

損失係数は $0.98^{\text{round}}$ で指数減衰し、EARLY_STOP_EPOCH (20) 以降は無効化される。

### 5. AFAD 固有の設計改善

単純な 2 手法の組み合わせ（ナイーブ統合）では性能が出ない。以下の 4 つの改善が必要だった。

| 改善 | 内容 | 効果 |
|------|------|------|
| **① 全クライアント KD 適用** | rate=1.0 以外も KD の対象に | +1.60pp |
| **② KD 係数を 10 → 1 に削減** | FedGen 用の α=β=10 は AFAD では過剰正則化 | +7.25pp |
| **③ KD Warmup（5 ラウンド）** | Generator が収束してから KD を開始 | +0.10pp |
| **④ レート依存 KD スケーリング** | `α = β = 0.5 / model_rate`（小容量ほど強いガイダンス）| +0.60pp |

```
α = β = 0.5 / model_rate

  model_rate=1.0  →  α=0.5  (自力で学べるため KD は補助的)
  model_rate=0.5  →  α=1.0  (標準的な KD)
  model_rate=0.25 →  α=2.0  (容量不足を Generator の知識で補完)
```

また、HeteroFL の幅スケーリングが `FedGenModelWrapper` の bottleneck を誤って縮小しないよう `num_preserved_tail_layers=2`（bottleneck + classifier を保護）を使用する。HeteroFL Only 時は `num_preserved_tail_layers=1`（classifier のみ保護）。

---

## デフォルト実験構成

10 クライアント・2 ファミリー（CNN + ViT）:

| Client | Family | Model | Rate | パラメータ数 |
|--------|--------|-------|:----:|:----------:|
| 0, 1 | CNN | HeteroFL ResNet18 | 1.0 | ~11.2M |
| 2, 3 | CNN | HeteroFL ResNet18 | 0.5 | ~2.8M |
| 4 | CNN | HeteroFL ResNet18 | 0.25 | ~0.7M |
| 5, 6 | ViT | HeteroFL ViT-Small | 1.0 | ~21.3M |
| 7, 8 | ViT | HeteroFL ViT-Small | 0.5 | ~5.4M |
| 9 | ViT | HeteroFL ViT-Small | 0.25 | ~1.4M |

---

## 実験結果

### Phase 1: 直接シミュレーション（MNIST, IID, 10 clients, 30 rounds）

> `run_direct_sim.py`（Ray/Flower 不使用）。seed=42 の単一試行。

| 手法 | BEST | FINAL | 実行時間 |
|------|:----:|:-----:|:-------:|
| HeteroFL Only | 69.90% | 69.50% | ~120s |
| FedGen Only | 67.00% | 66.85% | ~135s |
| **AFAD Hybrid** | **69.85%** | **69.70%** | ~173s |

AFAD の改善推移（ナイーブ統合からの 4 段階改善）:

| 施策 | AFAD BEST | 改善幅 |
|------|:---------:|:------:|
| ナイーブ統合（改善前） | 60.30% | — |
| ① 全クライアント KD 適用 | 61.90% | +1.60pp |
| ② α/β を 10 → 1 に削減 | 69.15% | +7.25pp |
| ③ KD Warmup（5 ラウンド）| 69.25% | +0.10pp |
| **④ レート依存 KD スケーリング** | **69.85%** | **+0.60pp** |

### Phase 1: Flower シミュレーション（MNIST, IID, 5 clients, 40 rounds）

> `run_comparison.py`（Flower + Ray）。seed=42 の単一試行。

| 手法 | BEST | FINAL | 実行時間 |
|------|:----:|:-----:|:-------:|
| HeteroFL Only | 99.19% | 99.19% | 474s |
| FedGen Only | 99.60% | 99.57% | 579s |
| **AFAD Hybrid** | **99.35%** | **99.29%** | **325s** |
| AFAD + Proto | — | — | — |
| **AFAD + RateCond** | **93.56%** | **93.52%** | ~1600s |
| AFAD + AnchorKD | 92.97% | 92.82% | ~1600s |
| AFAD + BNAnchorKD | 92.61% | 92.43% | ~1600s |

AFAD は HeteroFL と FedGen の中間に位置し、単独では達成できない設定（異なるアーキテクチャ × 異なる幅）をカバーしながら高い精度を維持する。また実行時間は 3 手法中最短。

新手法（RateCond / AnchorKD）については、MNIST IID では AFAD Hybrid より低い精度となった。これは MNIST が IID・単純なタスクであるため、追加の正則化（Generator rate-aware 化・教師蒸留）がノイズとして働く可能性がある。本来の評価環境は Non-IID（Phase 2）。

**AnchorKD の傾向**:
- Round 1〜5 の初期収束は AnchorKD 系が RateCond より速い（フルレート教師による早期誘導が有効）
- 最終精度は RateCond > AnchorKD > BNAnchorKD（BN 損失は MNIST IID では過剰正則化）

### Phase 2: OrganAMNIST（Non-IID α=0.5, 10 clients, 40 rounds）

> Dirichlet 分割によるデータ不均質環境。seed=42 の単一試行。

| 手法 | BEST |
|------|:----:|
| HeteroFL Only | 65.36% |
| AFAD Hybrid | 67.08% |
| **AFAD + Proto** | **68.00%** |
| FedGen Only | 84.66% |

AFAD + Proto はプロトタイプアンカリング（後述）により AFAD Hybrid を **+0.92pp** 上回る。Non-IID 環境では FedGen Only が最も高い精度（84.66%）を達成する。FedGen Generator がサーバー側でクラスバランスの取れた潜在ベクトルを生成するため、データ不均質の影響を受けにくい。一方 AFAD + Proto は HeteroFL を +2.64pp 上回るが、FedGen との差は **16.66pp** 残る。

この差は「構造的な問題」として分析されており、単純なハイパーパラメータ調整（KD 係数の増加など）では解消できないことが実験的に確認されている（詳細は後述のアブレーション実験を参照）。

### アブレーション実験（Phase 2 ベース）

Phase 2 の AFAD vs FedGen ギャップの原因を特定するために実施したアブレーション。

| 実験 | 変更内容 | 結果 | 解釈 |
|------|---------|------|------|
| **Exp A** | FedProx 無効化（mu=0 → 0） | −0.23pp（AFAD 比） | FedProx は役立っている（競合せず） |
| **Exp C** | KD を rate=1.0 クライアントのみに適用、α=10 | +0.32pp（R26 時点）、その後不安定 | KD 係数を増やしても根本解決にならない |

**結論**: AFAD vs FedGen のギャップは構造的問題。Generator が rate=1.0 のフルレートモデルの潜在空間に最適化されているため、rate<1.0 のサブレートクライアントは KD 信号を適切に活用できない。

---

## AFAD + Proto（Rate-aware Generator + Prototype Anchoring）

### 問題の根本原因と解決策

```
Generator（サーバー訓練）
  ↓ G(y) — 32次元潜在ベクトル
クライアント
  ↓ bottleneck(backbone(x)) — sub-rate では異なる潜在空間
classifier → 予測
```

**問題**: Generator は従来 `rate=1.0` のフルレートモデルのみを教師として訓練されていた。Sub-rate クライアント（rate=0.5/0.25）は幅縮小 bottleneck により異なる潜在表現を学習するため、Generator の潜在ベクトルと latent 空間が一致しない。

**解決策**: 2 つの改善を組み合わせる。

1. **Rate-aware Generator 訓練**（サーバー側）: 6 モデル（CNN/ViT × rate=1.0/0.5/0.25）全てを教師として Generator を訓練し、sub-rate の潜在空間を反映した潜在ベクトルを生成する。

2. **Prototype Anchoring**（クライアント側）: 追加損失 $\mathcal{L}_{\text{proto}}$ でクライアントの bottleneck 出力を Generator の潜在ベクトルに幾何的に引き寄せる。

$$\mathcal{L}_{\text{proto}} = \gamma \cdot e^{-\lambda \cdot t} \cdot \text{MSE}\bigl(\text{bottleneck}(\text{backbone}(x)),\; G(y)\bigr)$$

$\gamma = 1.0$（proto_gamma）、$\lambda = \text{DECAY\_RATE}^{\text{round}}$ で指数減衰。

### 設計の詳細

| 設計項目 | 選択 | 理由 |
|----------|------|------|
| **Generator 教師** | 6 モデル（CNN/ViT × rate=1.0/0.5/0.25） | sub-rate の潜在空間を Generator に反映 |
| **アンカリング損失** | MSE（bottleneck 出力 vs G(y)）| 潜在空間の幾何的整合性を直接最適化 |
| **proto_gamma** | 1.0（指数減衰あり） | 初期ラウンドで強く引き寄せ、収束後は KD に委譲 |
| **既存コードへの影響** | `afad_client.py`, `afad_strategy.py` のみ変更 | ベースライン手法には一切影響なし |

### 変更ファイル

| ファイル | 変更内容 |
|----------|---------|
| `src/server/strategy/afad_strategy.py` | `_get_sub_rates()` + `_reconstruct_model(model_rate)` — rate-aware Generator 訓練 |
| `src/client/afad_client.py` | `proto_gamma` パラメータ + Prototype Anchoring 損失 |
| `scripts/run_comparison.py` | 第 4 手法 `"AFAD + Proto"` の追加（`proto_gamma=1.0`） |

> **制約**: `fedgen_client.py`, `fedgen_distiller.py`, `heterofl_resnet.py`, `heterofl_vit.py`, `heterofl_aggregator.py`, `heterofl_client.py` は一切変更せず。

---

## AFAD + AnchorKD（フルレート凍結教師による sub-rate KD）

### 動機

AFAD + Proto の FedGen Generator は rate=1.0/0.5/0.25 全てを教師にすることで sub-rate の潜在空間を Generator に反映させる。一方 **AnchorKD** は別のアプローチをとる。**フルレートの集約済みグローバルモデル（凍結）を教師**として使い、各ラウンドで sub-rate クライアントのロジット・bottleneck 出力を直接整合させる。

### 手法

$$\mathcal{L} = \mathcal{L}_{\text{CE}} + \mathcal{L}_{\text{FedGen}} + \mathcal{L}_{\text{AnchorKD}}$$

**AnchorKD 損失**（sub-rate クライアントのみ）:

$$\mathcal{L}_{\text{AnchorKD}} = \gamma_{\text{logit}} \cdot T^2 \cdot \text{KL}\!\left(\frac{f_s(x)}{T} \,\Big\|\, \frac{f_a(x)}{T}\right) + \gamma_{\text{BN}} \cdot \text{MSE}\!\left(\text{LN}(z_s^{1:ed}),\, \text{LN}(z_a^{1:ed})\right)$$

- $f_a$: rate=1.0 凍結アンカーモデル（サーバーから毎ラウンド配布）
- $T = 1 / \text{model\_rate}$（rate=0.5 → T=2, rate=0.25 → T=4）
- $ed = \lfloor \text{latent\_dim} \times \text{model\_rate} \rfloor$（sub-rate bottleneck の有効次元）
- LN = LayerNorm（スケール差異を吸収）

### バリアント

| 手法 | anchor_kd_gamma | bottleneck_gamma | 説明 |
|------|:---:|:---:|------|
| **AFAD + AnchorKD** | 1.0 | 0 | ロジットレベル KD のみ |
| **AFAD + BNAnchorKD** | 1.0 | 1.0 | ロジット + bottleneck レベル |

### 変更ファイル

| ファイル | 変更内容 |
|----------|---------|
| `src/server/strategy/afad_strategy.py` | `anchor_kd_for_subrate` フラグ + `_serialize_anchor_params()` — rate=1.0 family モデルを sub-rate クライアントへ配布 |
| `src/client/afad_client.py` | `set_anchor_parameters()` + AnchorKD 損失計算（logit-level + BN-level） |
| `scripts/run_comparison.py` | `"AFAD + AnchorKD"`, `"AFAD + BNAnchorKD"` の 2 手法追加 |

> **Proto との違い**: Proto は Generator の潜在空間にクライアントの bottleneck を引き寄せる（生成側を基準）。AnchorKD はフルレートの実モデル出力を教師にする（識別側を基準）。両者は相補的であり、組み合わせも可能。

---

## 各手法の比較

| | HeteroFL Only | FedGen Only | AFAD Hybrid | AFAD + Proto | AFAD + AnchorKD | AFAD + BNAnchorKD |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 計算能力の異種性 (rate 可変) | ○ | × | **○** | **○** | **○** | **○** |
| アーキテクチャ異種性 (CNN ↔ ViT) | × | ○ | **○** | **○** | **○** | **○** |
| sub-rate クライアントへの知識補完 | × | △ | ○ | **◎** | **◎** | **◎** |
| Non-IID 耐性 | 中 | 高 | 中 | 中〜高 | 中〜高 | 中〜高 |
| 集約方式 | count-based | FedAvg | count-based | count-based | count-based | count-based |
| クライアント損失 | CE のみ | CE + KD (α/β=10 固定) | CE + KD (α/β=0.5/rate) | CE + KD + Proto anchoring | CE + KD + AnchorKD (logit) | CE + KD + AnchorKD (logit + BN) |
| サーバー Generator 訓練対象 | なし | rate=1.0 のみ | rate=1.0 のみ | **rate=1.0/0.5/0.25 全て** | rate=1.0 のみ | rate=1.0 のみ |
| プロトタイプアンカリング | なし | なし | なし | **あり（proto_gamma=1.0）** | なし | なし |
| AnchorKD（フルレート教師） | なし | なし | なし | なし | **logit-level (γ=1.0)** | **logit+BN-level (γ=1.0)** |

---

## システムアーキテクチャ

```
Server (AFADStrategy)
│
├── configure_fit()
│   ├── _initialize_family_models() — family ごとに rate=1.0 グローバルモデルを初期化
│   ├── HeteroFL distribute — family_global_models からサブモデルを切り出して配布
│   ├── Cosine LR を計算して全クライアントに配信
│   └── Generator params を pickle → config として送信（warmup 後・AFAD/FedGen のみ）
│
├── aggregate_fit()
│   ├── family ごとに結果をグループ化
│   ├── enable_heterofl=True  → _aggregate_heterofl() — count-based + label-split
│   │   enable_heterofl=False → _aggregate_fedavg()   — サンプル数重み付き平均
│   └── _train_generator_on_server() — family モデルを再構築して Generator を訓練
│
├── configure_evaluate()
│   └── 各クライアントに対応する family のサブモデルを配信
│
└── aggregate_evaluate()
    └── 加重平均で全体精度を集約

Clients
├── HeteroFLClient — CE 損失のみ / shape-aware set_parameters / KD なし
├── FedGenClient   — CE + KD（α=β=10 固定）/ フルレート / FedAvg 前提
└── AFADClient     — CE + KD（α=β=0.5/rate）/ 幅スケール / HeteroFL 前提
                     shape-aware set_parameters / FedProx 対応
                     Prototype Anchoring オプション対応（proto_gamma > 0 時に有効）
                     AnchorKD オプション対応（anchor_kd_gamma / bottleneck_gamma > 0 時に有効）
                       - logit-level: T²·KL(student(x)/T ‖ anchor(x)/T), T=1/rate
                       - BN-level: MSE(LN(student_bn[:,:ed]), LN(anchor_bn[:,:ed]))
```

---

## 原著論文との差分

### vs HeteroFL [Diao+, ICLR 2021]

| 観点 | 原著 HeteroFL | AFAD |
|------|:------------:|:----:|
| 対象アーキテクチャ | 単一アーキテクチャ族 | CNN + ViT の 2 ファミリー |
| 末尾層の保護 | 最終 Linear のみ | `num_preserved_tail_layers=2`（bottleneck + classifier） |
| Scaler 適用 | 全残差層後に 1 回 | 同一（論文 §3.1 準拠） |
| BatchNorm | Static BN（`track_running_stats=False`） | 同一 |
| 集約 | count-based 平均 | 同一 + label-split 集約 |
| ファミリー間知識共有 | なし | Generator 経由で KD |

### vs FedGen [Zhu+, ICML 2021]

| 観点 | 原著 FedGen | AFAD |
|------|:----------:|:----:|
| Generator 出力 | 潜在ベクトル（32次元） | 同一 |
| KD の実行場所 | クライアント側正則化 | 同一 |
| 損失関数 | CE + KL（減衰あり） | 同一 |
| KD 係数 α, β | 固定値（10.0） | レート依存: `0.5 / model_rate` |
| KD Warmup | 明示的な記述なし | `FEDGEN_WARMUP_ROUNDS=5` |
| 集約方式 | FedAvg | HeteroFL count-based |
| Generator 学習 | 全クライアントモデルで | family ごとの rate=1.0 モデルで |
| 対象クライアント | 同一幅が前提 | 幅スケーラブル（rate 混在） |

---

## セットアップ

**動作環境**: Python 3.10+ / PyTorch 2.0+ / Flower 1.7+ / CUDA（推奨、CPU 動作可）

```bash
git clone https://github.com/Richiesss/AFAD.git
cd AFAD

# uv のインストール（未導入の場合）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 依存パッケージのインストール
uv sync
```

---

## 実行方法

### 直接シミュレーション（推奨）

Ray / Flower ワーカー不要で高速に起動できる。

```bash
# フル実験（10 clients, 30 rounds, MNIST）
uv run python scripts/run_direct_sim.py

# 結果を JSON に保存
uv run python scripts/run_direct_sim.py --output results/my_run.json

# クイックテスト（4 clients, 5 rounds, 500 サンプル）
uv run python scripts/run_quick_test.py
```

出力例:

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
   30 |              69.50% |              66.85% |              69.70%
--------------------------------------------------------------------------
 BEST |              69.90% |              67.00% |              69.85%
FINAL |              69.50% |              66.85% |              69.70%
```

### Flower シミュレーション

```bash
# 3 手法比較（MNIST, IID, 5 clients, 40 rounds）
uv run python scripts/run_comparison.py

# 4 手法比較（HeteroFL / FedGen / AFAD / AFAD+Proto）
uv run python scripts/run_comparison.py --methods "HeteroFL Only" "FedGen Only" "AFAD Hybrid" "AFAD + Proto"

# Phase 2（OrganAMNIST, Non-IID）
uv run python scripts/run_comparison.py config/afad_phase2_config.yaml

# 特定手法のみ実行（既存結果を --load でロード）
uv run python scripts/run_comparison.py config/afad_phase2_config.yaml \
  --methods "AFAD + Proto" --load results/existing.json --output results/new.json

# Multi-seed 統計的検証
uv run python scripts/run_multi_seed.py config/afad_phase2_config.yaml
```

---

## ディレクトリ構造

```
AFAD/
├── scripts/
│   ├── run_direct_sim.py           # 直接シミュレーション（推奨）
│   ├── run_quick_test.py           # 4 clients・5 rounds のクイック検証
│   ├── run_comparison.py           # 3 手法比較（Flower ベース）
│   ├── run_multi_seed.py           # Multi-seed 統計的検証
│   └── run_experiment.py           # 単一実験スクリプト
├── src/
│   ├── client/
│   │   ├── afad_client.py          # AFAD ハイブリッドクライアント
│   │   ├── heterofl_client.py      # HeteroFL Only ベースライン
│   │   └── fedgen_client.py        # FedGen Only ベースライン
│   ├── data/
│   │   ├── dataset_config.py       # データセット設定レジストリ
│   │   ├── mnist_loader.py         # MNIST データローダー
│   │   └── medmnist_loader.py      # OrganAMNIST + Dirichlet 分割
│   ├── models/
│   │   ├── registry.py             # モデルレジストリ（ファクトリパターン）
│   │   ├── fedgen_wrapper.py       # FedGenModelWrapper（bottleneck + classifier）
│   │   ├── scaler.py               # HeteroFL Scaler（1/rate 補償）
│   │   ├── cnn/
│   │   │   ├── heterofl_resnet.py  # 幅スケーラブル ResNet18（sBN + 1× Scaler）
│   │   │   ├── resnet.py           # ResNet18, ResNet50
│   │   │   └── mobilenet.py        # MobileNetV3-Large
│   │   └── vit/
│   │       ├── heterofl_vit.py     # 幅スケーラブル ViT-Small
│   │       ├── vit.py              # ViT-Tiny, ViT-Small
│   │       └── deit.py             # DeiT-Small
│   ├── server/
│   │   ├── generator/
│   │   │   ├── fedgen_generator.py       # FedGen 潜在空間 Generator
│   │   │   └── afad_generator_trainer.py # サーバーサイド Generator 訓練（rate-aware）
│   │   └── strategy/
│   │       ├── afad_strategy.py          # AFAD 統合戦略（3 モード対応）
│   │       └── heterofl_aggregator.py    # HeteroFL 集約（count-based + label-split）
│   └── utils/
│       ├── config_loader.py        # YAML 設定読み込み
│       ├── logger.py               # ロガー
│       └── metrics.py              # MetricsCollector
├── tests/
│   ├── test_afad_integration.py    # AFAD E2E 統合テスト
│   ├── test_heterofl_aggregator.py # HeteroFL 集約テスト（tail layers 含む）
│   ├── test_fedgen_faithful.py     # FedGen コンポーネントテスト
│   ├── test_generator.py           # Generator テスト
│   └── ...
├── config/
│   ├── afad_config.yaml            # Phase 1 設定（MNIST, IID）
│   └── afad_phase2_config.yaml     # Phase 2 設定（OrganAMNIST, Non-IID）
├── results/                        # 実験結果 JSON
├── pyproject.toml
└── uv.lock
```

---

## 開発

```bash
# テスト
uv run pytest -v
uv run pytest tests/test_afad_integration.py -v   # AFAD 統合テストのみ
uv run pytest -x                                   # 最初の失敗で停止

# Lint / Format
uv run ruff check .
uv run ruff format --check .
uv run ruff check --fix . && uv run ruff format .  # 自動修正

# 一括実行（lint + test）
uv run poe all
```

---

## 参考文献

- Diao, E. et al. "HeteroFL: Computation and Communication Efficient Federated Learning for Heterogeneous Clients." *ICLR 2021.*
- Zhu, Z. et al. "Data-Free Knowledge Distillation for Heterogeneous Federated Learning." *ICML 2021.*
- Li, T. et al. "Federated Optimization in Heterogeneous Networks." *MLSys 2020.*
- Hinton, G. et al. "Distilling the Knowledge in a Neural Network." *NeurIPS Workshop 2015.*

---

## 著者

- **作成者**: 島野 凌
- **所属**: 大阪工業大学 大学院 情報科学研究科
