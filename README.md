# AFAD: Adaptive Federated Architecture Distribution

> **連合学習における二重の異種性（計算能力 × モデルアーキテクチャ）を同時に解決するハイブリッド連合学習フレームワーク**

---

## Abstract

実際の連合学習（Federated Learning; FL）環境では、クライアントは計算能力とモデルアーキテクチャの両面で異質である。既存手法は一方の問題のみを扱う：HeteroFL [Diao+, ICLR 2021] は幅スケーリングで計算能力の異種性に対応するが、異なるアーキテクチャ族（CNN と Transformer など）間の知識共有を行わない。FedGen [Zhu+, ICML 2021] はデータフリー知識蒸留（KD）でアーキテクチャ異種性に対応するが、全クライアントが同一サイズのモデルを持つことを前提とする。

本研究では **AFAD（Adaptive Federated Architecture Distribution）** を提案する。AFAD は両手法を統合し、共有 32 次元潜在空間と Generator を介してアーキテクチャをまたいだ知識共有を実現しながら、幅スケーリングで計算能力の異種性にも対応する。単純な統合（ナイーブ統合）から 4 段階の改善を経て、さらに **Prototype Anchoring**・**Rate-conditioned Generator**・**AnchorKD** という 3 種の拡張手法を提案する。

Non-IID 環境（OrganAMNIST, α=0.5）において、AFAD + Proto は HeteroFL Only を **+2.64pp** 上回る。また、sub-rate クライアントの潜在空間不整合という構造的問題を特定し、**FedAvg 集約・サーバー側蒸留・Nested Bottleneck** という 3 種のギャップ解消アプローチを体系的に検証する。全アプローチがベースラインを下回った実験結果を通じて、FedGen との性能差が単一手法では解消できない複合的な構造問題であることを示す。

---

## 目次

1. [問題設定](#1-問題設定)
2. [関連研究](#2-関連研究)
3. [手法：AFAD](#3-手法afad)
   - 3.1 [共有潜在空間設計](#31-共有潜在空間fedgenmodelwrapper)
   - 3.2 [HeteroFL 統合](#32-heterofl統合)
   - 3.3 [FedGen 統合](#33-fedgen統合)
   - 3.4 [ナイーブ統合から AFAD Hybrid へ](#34-ナイーブ統合からafad-hybridへ)
4. [拡張手法](#4-拡張手法)
   - 4.1 [AFAD + Proto（Prototype Anchoring）](#41-afad--proto)
   - 4.2 [AFAD + RateCond（Rate-conditioned Generator）](#42-afad--ratecond)
   - 4.3 [AFAD + AnchorKD（フルレート凍結教師）](#43-afad--anchorkd)
5. [実験](#5-実験)
   - 5.1 [実験設定](#51-実験設定)
   - 5.2 [Phase 1: MNIST IID](#52-phase-1-mnist-iid)
   - 5.3 [Phase 2: OrganAMNIST Non-IID](#53-phase-2-organamnist-non-iid)
     - 5.3.1 [全手法比較](#531-全手法比較)
     - 5.3.2 [FedGen ギャップ解消実験](#532-fedgen-ギャップ解消実験)
   - 5.4 [アブレーション実験](#54-アブレーション実験)
6. [分析・考察](#6-分析考察)
7. [各手法の比較表](#7-各手法の比較表)
8. [システムアーキテクチャ](#8-システムアーキテクチャ)
9. [原著論文との差分](#9-原著論文との差分)
10. [セットアップ・実行方法](#10-セットアップ実行方法)
11. [ディレクトリ構造](#11-ディレクトリ構造)
12. [参考文献](#12-参考文献)

---

## 1. 問題設定

### 1.1 二重の異種性

実際の FL 環境には、同時に 2 種類の異種性が存在する。

| 異種性の種類 | 具体例 | 従来手法の限界 |
|-------------|--------|--------------|
| **計算能力の異種性** | スマートフォン vs エッジサーバー | FedGen は全員が同一サイズのモデルを前提とする |
| **アーキテクチャの異種性** | CNN（ResNet）vs Transformer（ViT） | HeteroFL はファミリー間の知識共有を行わない |

この 2 種類の異種性を**同時に**扱える手法は、AFAD 以前には存在しなかった。

### 1.2 具体的な設定

本研究では以下の設定を想定する：

- **10 クライアント**、**2 ファミリー**（CNN: HeteroFL ResNet18、ViT: HeteroFL ViT-Small）
- 各ファミリー内に rate=1.0 / 0.5 / 0.25 の 3 段階の幅スケーリングを持つクライアントが混在
- データ分布は IID（Phase 1）および Non-IID Dirichlet 分割（Phase 2、α=0.5）

| Client | Family | Model | Rate | パラメータ数 |
|:------:|:------:|-------|:----:|:---------:|
| 0, 1 | CNN | HeteroFL ResNet18 | 1.0 | ~11.2M |
| 2, 3 | CNN | HeteroFL ResNet18 | 0.5 | ~2.8M |
| 4 | CNN | HeteroFL ResNet18 | 0.25 | ~0.7M |
| 5, 6 | ViT | HeteroFL ViT-Small | 1.0 | ~21.3M |
| 7, 8 | ViT | HeteroFL ViT-Small | 0.5 | ~5.4M |
| 9 | ViT | HeteroFL ViT-Small | 0.25 | ~1.4M |

### 1.3 研究目標

> **RQ**: 計算能力・アーキテクチャの二重異種性が存在する FL 環境で、全クライアントの精度を最大化するための効果的な知識共有・集約戦略はどのようなものか？

特に以下の問いに答えることを目標とする：
- **RQ1**: HeteroFL と FedGen の単純な統合（ナイーブ統合）はなぜ機能しないのか？
- **RQ2**: sub-rate クライアント（rate<1.0）が FedGen の KD 信号を適切に活用できない根本原因は何か？
- **RQ3**: 各拡張手法（Proto / RateCond / AnchorKD）は Non-IID 環境での性能をどの程度改善するか？

---

## 2. 関連研究

### 2.1 計算能力の異種性への対応

**HeteroFL** [Diao et al., ICLR 2021] は、グローバルモデルの先頭チャネルを rate に応じて切り出したサブモデルをクライアントに配布することで、異なる計算能力のクライアントが同一の FL に参加できる仕組みを提供する。Static BatchNorm と Scaler を組み合わせることで、幅縮小による活性化値のスケール変化を補償する。

**FjORD** [Horvath et al., NeurIPS 2021] は Ordered Dropout により、HeteroFL と類似の幅スケーリングを実現するが、Static BatchNorm は使用しない。

### 2.2 アーキテクチャの異種性への対応

**FedGen** [Zhu et al., ICML 2021] は、サーバーサイドで Generator $G: y \to z$ を学習し、クライアントに生成された潜在ベクトルを用いた KD を適用することで、データフリーかつアーキテクチャに依存しない知識共有を実現する。ただし、全クライアントが同一サイズのモデルを保持することを前提とする。

**DS-FL** [Goetz et al., 2020] はサーバーがプロキシデータを用いた蒸留を行うが、プロキシデータへのアクセスを必要とする。

**FedDF** [Lin et al., NeurIPS 2020] は蒸留ベースのモデルフュージョンを提案するが、同様にサーバー側データを要求する。

### 2.3 本研究の位置づけ

AFAD は計算能力・アーキテクチャ両方の異種性に対応し、かつサーバー側の学習データを一切使用しない（Data-Free）点で、既存手法に対して明確なアドバンテージを持つ。

---

## 3. 手法：AFAD

### 3.1 共有潜在空間（FedGenModelWrapper）

異なるアーキテクチャ・幅のモデルが同一の Generator と知識を共有するために、全モデルに共通の **32 次元ボトルネック層**を導入する。

```
入力 x
  → backbone（幅スケーラブル）  ← HeteroFL の width-scaling 対象
  → bottleneck（固定: feature_dim → 32）   ← HeteroFL では保護（num_preserved_tail_layers=2）
  → classifier（固定: 32 → num_classes）   ← HeteroFL では保護
```

`FedGenModelWrapper` がこの構造を全モデルに付加する。backbone のアーキテクチャ（ResNet / ViT）や幅（rate=1.0/0.5/0.25）によらず、classifier 直前の表現は必ず 32 次元に統一される。

**設計上の重要事項**: HeteroFL の幅スケーリングが bottleneck を誤って縮小しないよう `num_preserved_tail_layers=2`（bottleneck + classifier を保護）を設定する。HeteroFL Only の場合は `num_preserved_tail_layers=1`（classifier のみ保護）。

全体の構造を図示すると：

```
CNN クライアント (HeteroFL ResNet18)     ViT クライアント (HeteroFL ViT-Small)
  rate=1.0: 512ch                          rate=1.0: 384dim
  rate=0.5: 256ch                          rate=0.5: 192dim
  rate=0.25: 128ch                         rate=0.25: 96dim
      │                                          │
      │  backbone（幅可変）                        │  backbone（幅可変）
      ↓                                          ↓
  bottleneck（→ 32次元）                bottleneck（→ 32次元）
      │                                          │
      └──────────── 共有潜在空間 32次元 ────────────┘
                            │
                     FedGen Generator G
                    （サーバーで訓練）
```

### 3.2 HeteroFL 統合

各クライアントの `model_rate`（1.0 / 0.5 / 0.25）に応じて、グローバルモデルの先頭チャネルを切り出したサブモデルを配布する。

**主要コンポーネント**:

| コンポーネント | 内容 | 目的 |
|--------------|------|------|
| **Static BatchNorm** | `track_running_stats=False` | 幅の異なるクライアント間での running stats 不整合を防止 |
| **Scaler** | 出力 ÷ model_rate（全残差層後に 1 回） | 幅縮小による活性化値スケール変化を補償 |
| **count-based 集約** | 更新カウントで重み付き平均 | FedAvg のサンプル数重みより安定した集約 |
| **label-split 集約** | 出力層の各行を担当クラスのクライアントのみで更新 | Non-IID 環境での出力層のクラス偏りを防止 |

### 3.3 FedGen 統合

サーバーで Generator $G$ を学習し、クライアントに KD を行う。

**Generator 学習損失**:

$$\mathcal{L}_G = \alpha_G \cdot \mathcal{L}_{\text{teacher}} + \eta \cdot \mathcal{L}_{\text{diversity}}$$

- $\mathcal{L}_{\text{teacher}}$: 全ファミリーモデルの `forward_from_latent(G(y))` による加重 CE 損失
- $\mathcal{L}_{\text{diversity}}$: Generator のモード崩壊を防止する多様性損失

**クライアント側 KD 損失（AFAD Hybrid）**:

$$\mathcal{L}_{\text{AFAD}} = \underbrace{\text{CE}(f(x), y)}_{\text{予測損失}} + \alpha \cdot \underbrace{\text{CE}\!\left(\text{cls}(G(y_{\text{rand}})),\, y_{\text{rand}}\right)}_{\text{教師損失}} + \beta \cdot \underbrace{\text{KL}\!\left(f(x) \,\|\, \text{cls}(G(y_{\text{real}}))\right)}_{\text{潜在マッチング損失}}$$

損失係数は $0.98^{\text{round}}$ で指数減衰し、EARLY_STOP_EPOCH（20 エポック）以降は無効化される。

### 3.4 ナイーブ統合から AFAD Hybrid へ

HeteroFL と FedGen を単純に結合するだけでは性能が大幅に低下する（ナイーブ統合: 60.30%）。以下の 4 段階の改善により性能を回復した。

| 施策 | 内容 | AFAD BEST | 改善幅 |
|------|------|:---------:|:------:|
| ナイーブ統合（改善前） | HeteroFL + FedGen を素朴に組み合わせる | 60.30% | — |
| **① 全クライアント KD 適用** | rate<1.0 のクライアントも KD の対象に含める | 61.90% | +1.60pp |
| **② KD 係数を 10 → 適正値に削減** | FedGen の α=β=10 は AFAD では過剰正則化 | 69.15% | +7.25pp |
| **③ KD Warmup（5 ラウンド）** | Generator が収束してから KD を開始 | 69.25% | +0.10pp |
| **④ レート依存 KD スケーリング** | `α = β = 0.5 / model_rate` | **69.85%** | **+0.60pp** |

> **④ の設計根拠**: 小容量クライアントほど Generator の知識に依存する必要があるため、model_rate に反比例させて KD 係数を設定する。
>
> ```
> α = β = 0.5 / model_rate
>   rate=1.0  → α=0.5  （自力で学べるため KD は補助的）
>   rate=0.5  → α=1.0  （標準的な KD）
>   rate=0.25 → α=2.0  （容量不足を Generator の知識で補完）
> ```

---

## 4. 拡張手法

### 4.1 AFAD + Proto

#### 問題の特定

Phase 2（Non-IID）の実験で、AFAD Hybrid と FedGen Only の間に **16.66pp** の大きなギャップが存在することが判明した。このギャップの原因を特定するためにアブレーション実験を実施した結果、**Generator が rate=1.0 のフルレートモデルのみを教師として訓練されているため、sub-rate クライアントの潜在空間と Generator の潜在ベクトルが一致しない**という構造的問題が根本原因であると結論付けた。

```
Generator（サーバー訓練）
  ↓ G(y) — rate=1.0 の潜在空間に特化した 32 次元ベクトル
クライアント（sub-rate）
  ↓ bottleneck(backbone(x)) — 幅縮小により異なる潜在表現
classifier → 予測（KD 信号が活用できない）
```

#### 解決策：2 段階のアプローチ

**1. Rate-aware Generator 訓練**（サーバー側）:

6 モデル（CNN/ViT × rate=1.0/0.5/0.25）全てを教師として Generator を訓練し、sub-rate クライアントの潜在空間を Generator が反映できるようにする。

**2. Prototype Anchoring**（クライアント側）:

追加損失 $\mathcal{L}_{\text{proto}}$ でクライアントの bottleneck 出力を Generator の潜在ベクトルに幾何的に引き寄せる。

$$\mathcal{L}_{\text{proto}} = \gamma \cdot \delta^t \cdot \text{MSE}\!\left(\text{bottleneck}(\text{backbone}(x)),\; G(y)\right)$$

- $\gamma = 1.0$（proto_gamma）
- $\delta^t = 0.98^{\text{round}}$ で指数減衰（初期ラウンドで強く引き寄せ、収束後は KD に委譲）

#### 設計の詳細

| 設計項目 | 選択 | 理由 |
|---------|------|------|
| Generator 教師 | 6 モデル（CNN/ViT × rate=1.0/0.5/0.25） | sub-rate の潜在空間を Generator に反映 |
| アンカリング損失 | MSE（bottleneck 出力 vs G(y)） | 潜在空間の幾何的整合性を直接最適化 |
| proto_gamma | 1.0（指数減衰あり） | 初期ラウンドで強く引き寄せ、収束後は KD に委譲 |

### 4.2 AFAD + RateCond

AFAD + Proto の Rate-aware Generator 訓練をさらに発展させ、Generator 自体が rate 情報を条件として潜在ベクトルを生成する構造に拡張する。

**Rate-conditioned Generator** $G(y, r)$:

$$z = G(y, r) = \text{Trunk}(y) + \text{Head}_r(y)$$

- **Trunk**: 全 rate に共通の特徴抽出
- **Head_r**: rate ごとのヘッド（rate=1.0/0.5/0.25 の 3 種）
- **条件付け**: 離散 rate embedding + $\log(\text{rate})$ スカラーを concat

rate=1.0 ヘッドは既存の Generator の重みで初期化することで、退行リスクを最小化する。

**クライアント側の変更**: Generator 呼び出しを `G(y)` → `G(y, rate=self.model_rate)` に変更し、各クライアントが自身の model_rate に対応した潜在ベクトルを受け取る。

### 4.3 AFAD + AnchorKD

#### 動機

Proto/RateCond がサーバーの Generator 側から sub-rate の潜在不整合を解消しようとするのに対し、**AnchorKD** は識別側（クライアント側）から直接解消するアプローチをとる。**フルレートの集約済みグローバルモデル（凍結）を教師（アンカー）**として使い、sub-rate クライアントのロジット・bottleneck 出力を直接整合させる。

#### 手法

$$\mathcal{L} = \mathcal{L}_{\text{CE}} + \mathcal{L}_{\text{FedGen}} + \mathcal{L}_{\text{AnchorKD}}$$

**AnchorKD 損失**（sub-rate クライアントのみ、$\text{rate} < 1.0$）:

$$\mathcal{L}_{\text{AnchorKD}} = \gamma_{\text{logit}} \cdot T^2 \cdot \text{KL}\!\left(\frac{f_s(x)}{T} \,\Big\|\, \frac{f_a(x)}{T}\right) + \gamma_{\text{BN}} \cdot \text{MSE}\!\left(\text{LN}(z_s^{1:ed}),\, \text{LN}(z_a^{1:ed})\right)$$

| 記号 | 定義 |
|------|------|
| $f_a$ | rate=1.0 凍結アンカーモデル（サーバーから毎ラウンド配布） |
| $f_s$ | sub-rate 学習中の student モデル |
| $T$ | 温度 $= 1 / \text{model\_rate}$（rate=0.5 → T=2、rate=0.25 → T=4） |
| $ed$ | $\lfloor \text{latent\_dim} \times \text{model\_rate} \rfloor$（sub-rate bottleneck の有効次元） |
| $\text{LN}$ | LayerNorm（スケール差異を吸収） |
| $z_s, z_a$ | student・anchor それぞれの bottleneck 出力 |

**温度 T の設計根拠**: 容量の小さいクライアントほど教師の出力分布を「ソフト」にすることで、模倣しやすくなる。$T = 1/\text{rate}$ とすることで、容量と温度が自動的に対応する。

#### バリアント

| 手法 | $\gamma_{\text{logit}}$ | $\gamma_{\text{BN}}$ | 説明 |
|------|:---:|:---:|------|
| **AFAD + AnchorKD** | 1.0 | 0 | ロジットレベル KD のみ |
| **AFAD + BNAnchorKD** | 1.0 | 1.0 | ロジット + bottleneck レベル |

#### Proto との対比

| 観点 | AFAD + Proto | AFAD + AnchorKD |
|------|-------------|----------------|
| 整合の基準 | 生成側（Generator 潜在空間） | 識別側（フルレートモデルの出力） |
| 教師の更新 | 毎ラウンド Generator が更新 | 毎ラウンドフルレートモデルが更新 |
| 送信コスト | Generator パラメータ | フルレートモデルパラメータ（大） |
| 相補性 | 両者を組み合わせることも可能 | ← |

---

## 5. 実験

### 5.1 実験設定

#### データセット

| データセット | 分割 | クライアント数 | ラウンド数 | 用途 |
|------------|------|:-----------:|:--------:|------|
| **MNIST** | IID | 5 | 40 | Phase 1: 動作確認・手法比較 |
| **OrganAMNIST** | Non-IID (Dirichlet α=0.5) | 10 | 40 | Phase 2: 本番評価 |

#### 共通設定

| ハイパーパラメータ | 値 |
|-----------------|---|
| Optimizer | SGD（momentum=0.9, weight_decay=1e-4） |
| Learning Rate | Cosine Annealing（初期 0.01、最終 0.0001） |
| Local Epochs | 3 |
| Batch Size | 64 |
| FedProx μ | 0.01 |
| Generator noise_dim / latent_dim | 32 |
| Generator hidden_dim | 256 |
| KD Warmup | 5 ラウンド |
| EARLY_STOP_EPOCH | 20 |

#### ベースライン

- **HeteroFL Only**: FedGen なし、CE 損失のみ
- **FedGen Only**: HeteroFL なし、全クライアント rate=1.0

### 5.2 Phase 1: MNIST IID

#### 5.2.1 直接シミュレーション（10 clients, 30 rounds）

> `run_direct_sim.py`（Ray/Flower 不使用）。seed=42 の単一試行。

| 手法 | BEST | FINAL | 時間 |
|------|:----:|:-----:|:---:|
| HeteroFL Only | 69.90% | 69.50% | ~120s |
| FedGen Only | 67.00% | 66.85% | ~135s |
| **AFAD Hybrid** | **69.85%** | **69.70%** | ~173s |

AFAD の 4 段階改善の詳細（直接シミュレーション）:

| 施策 | AFAD BEST | 改善幅 |
|------|:---------:|:------:|
| ナイーブ統合 | 60.30% | — |
| ① 全クライアント KD 適用 | 61.90% | +1.60pp |
| ② α/β を 10 → 1 に削減 | 69.15% | +7.25pp |
| ③ KD Warmup（5 ラウンド） | 69.25% | +0.10pp |
| **④ レート依存 KD スケーリング** | **69.85%** | **+0.60pp** |

#### 5.2.2 Flower シミュレーション（5 clients, 40 rounds）

> `run_comparison.py`（Flower + Ray）。seed=42 の単一試行。

| 手法 | BEST | FINAL | 時間 |
|------|:----:|:-----:|:---:|
| HeteroFL Only | 99.19% | 99.19% | 474s |
| FedGen Only | 99.60% | 99.57% | 579s |
| AFAD Hybrid | 99.35% | 99.29% | 325s |
| AFAD + Proto | — | — | — |
| **AFAD + RateCond** | **93.56%** | **93.52%** | ~1600s |
| AFAD + AnchorKD | 92.97% | 92.82% | ~1600s |
| AFAD + BNAnchorKD | 92.61% | 92.43% | ~1600s |

> **注**: MNIST IID では RateCond / AnchorKD 系が AFAD Hybrid を下回る。これは IID・単純なタスクでは追加正則化がノイズとなるためと考えられる。本来の評価は Non-IID（Phase 2）で行う。

**AnchorKD の初期収束優位性**（MNIST IID、Round 1〜5）:

| Round | AFAD + RateCond | AFAD + AnchorKD | AFAD + BNAnchorKD |
|:-----:|:---------------:|:---------------:|:-----------------:|
| 1 | 9.59% | 9.26% | 9.18% |
| 2 | 73.34% | 75.21% | 73.47% |
| 3 | 81.40% | 82.50% | 84.27% |
| 4 | 84.64% | 86.12% | 87.65% |
| 5 | 89.03% | 88.78% | 89.27% |

AnchorKD 系は序盤（Round 3〜4）でフルレート教師による早期誘導が有効に働き、RateCond より速い収束を示す。

### 5.3 Phase 2: OrganAMNIST Non-IID

> Dirichlet 分割（α=0.5）による Non-IID 環境。10 clients、40 rounds、seed=42 の単一試行。

#### 5.3.1 全手法比較

| 手法 | BEST | Final |
|------|:----:|:-----:|
| HeteroFL Only | 65.36% | 64.90% |
| AFAD Hybrid | 67.84% | 67.73% |
| AFAD + AnchorKD | 66.97% | 66.69% |
| AFAD + BNAnchorKD | 66.79% | 66.79% |
| AFAD + RateCond | 67.94% | 67.68% |
| **AFAD + Proto** | **68.00%** | **67.39%** |
| FedGen Only | 84.66% | 84.62% |

AFAD + Proto は AFAD Hybrid を **+0.16pp**、HeteroFL Only を **+2.64pp** 上回り、AFAD 系で最良の結果を示す。

しかし FedGen Only（84.66%）との差は **16.66pp** に達する。FedGen Only が Non-IID で強い理由は、サーバーサイドの Generator がクラスバランスの取れた潜在ベクトルを生成することでクライアントのデータ偏りの影響を無効化できるためである。

#### 5.3.2 FedGen ギャップ解消実験

16.66pp のギャップを縮めるために 3 つのアプローチを体系的に検討した。各手法はそれぞれ独立したブランチで実装・実験した。

**Case C: FedAvg 集約（feature/fedavg-aggregation）**

HeteroFL の count-based 集約を、FedGen と同じサンプル数重み付けの FedAvg に置き換える。Sub-rate パラメータは対応するスライスにのみ加算する shape-aware な実装とした。

**Case A: Server-side Distillation（feature/server-side-distillation）**

集約後、サーバー上で Generator を用いてクラスバランスの取れた潜在ベクトルを生成し、各 family グローバルモデルを追加学習（20 steps, Adam, lr=1e-4）。Non-IID によるデータ偏りをサーバー側で補正することを狙う。

**Case B: Nested Bottleneck（feature/nested-bottleneck）**

Sub-rate クライアントは bottleneck weight の最初の `int(32 × rate)` 行のみを所有し、階層的な共有部分空間を形成する。rate=0.5 クライアントは 16 次元、rate=0.25 クライアントは 8 次元の有効潜在空間を保持する。

**実験結果**

| 手法 | BEST | Final | vs AFAD Hybrid |
|------|:----:|:-----:|:--------------:|
| AFAD Hybrid（ベースライン） | 67.84% | 67.73% | — |
| AFAD + ServerDistill (Case A) | 67.14% | 66.63% | **−0.70pp** |
| AFAD + NestedBN (Case B) | 62.53% | 60.97% | **−4.76pp** |
| AFAD + FedAvg (Case C) | 59.35% | 58.53% | **−8.20pp** |
| FedGen Only（上限） | 84.66% | 84.62% | +16.82pp |

**結論**: 3 手法ともベースラインを下回る結果となり、ギャップの解消には至らなかった。

- **Case C（FedAvg 集約）**: HeteroFL の count-based 集約はサブレートモデル専用に設計されており、単純なサンプル重み付けは構造的不整合を生む（−8.20pp）
- **Case A（Server Distillation）**: Generator がクラスバランス出力に過学習（loss→0.0000）し、Non-IID で形成された局所特化パターンを上書きしてしまう（−0.70pp）
- **Case B（Nested Bottleneck）**: 階層的制約が学習を阻害し、特に early rounds での収束が遅れる（−4.76pp）

FedGen Only との **16.66pp ギャップは付加的な手法では解消できない構造的問題**であることが確認された。根本解決には、HeteroFL の width-scaling 制約を緩和しつつ FedGen の Non-IID 耐性を活かす、より抜本的なアーキテクチャ変更が必要と考えられる。

### 5.4 アブレーション実験

Phase 2（Non-IID）をベースに実施。

| 実験 | 変更内容 | 結果 | 解釈 |
|------|---------|------|------|
| **Exp A** | FedProx 無効化（μ=0.01 → 0） | −0.23pp | FedProx は Non-IID 安定性に寄与（競合せず） |
| **Exp C** | KD を rate=1.0 のみ・α=10 | +0.32pp（Round 26 時点）→ 不安定 | KD 係数増加は根本解決にならない |
| **Case A** | Server-side Distillation (steps=20) | −0.70pp | Generator 過学習が Non-IID 局所特化を破壊 |
| **Case B** | Nested Bottleneck (8/16 dim) | −4.76pp | 階層的制約が学習を阻害 |
| **Case C** | FedAvg 集約（shape-aware） | −8.20pp | HeteroFL count-based 集約の合理性を逆説的に確認 |

**結論**: AFAD vs FedGen のギャップは構造的問題。Generator が rate=1.0 の潜在空間に特化しているため、sub-rate クライアントは KD 信号を適切に活用できない。単純なハイパーパラメータ調整・集約方式変更・潜在空間制約のいずれも根本解決にならない。

---

## 6. 分析・考察

### 6.1 ナイーブ統合が失敗する理由

HeteroFL と FedGen を素朴に統合すると **9.55pp（60.30% → 69.85%）もの性能差**が生じる。その主な原因は以下の 2 点：

1. **過剰正則化**: FedGen の KD 係数（α=β=10）は同一幅モデルを前提に設計されており、幅の異なるモデルに適用すると正則化が過剰になる（改善幅の 76%、7.25pp を占める）。
2. **rate=1.0 クライアントへの KD 制限**: 元の FedGen の実装では rate=1.0 のクライアントのみを KD 対象とすることで精度を守っていたが、AFAD では sub-rate クライアントも学習する必要があるため、全クライアントへの KD 適用が必要。

### 6.2 FedGen vs AFAD の構造的ギャップ

Non-IID 環境での FedGen（84.66%）と AFAD + Proto（68.00%）の 16.66pp のギャップは、以下の構造的な問題に起因する：

1. **潜在空間の不整合**: Generator は rate=1.0 の bottleneck に最適化されており、rate=0.5/0.25 のクライアントの bottleneck 出力とは異なる分布を持つ。KD 信号が「ノイズ」として働く。

2. **集約方式の非対称性**: FedGen は FedAvg（サンプル数重み）を使用し、Non-IID 下でも比較的安定した集約ができる。AFAD は count-based 集約（HeteroFL 由来）を使用しており、Non-IID 下での挙動が異なる。**ただし、Case C の実験（FedAvg に切り替え）でむしろ精度が低下（−8.20pp）したことから、HeteroFL の count-based 集約はサブレートモデルへの適切なパラメータ分配に不可欠であることが示された。**

3. **FedGen の本質的な Non-IID 耐性**: FedGen の Generator はサーバー側でクラスバランスの取れたデータを生成するため、クライアントのデータ偏りを補正できる。AFAD の HeteroFL 集約はこの恩恵を完全には受けられない。**Case A の実験（サーバー側蒸留）では Generator 過学習（loss→0.0000）が Non-IID の局所特化を破壊した。**

**ギャップ解消実験の総括**: 3 つの解消アプローチ（FedAvg 集約・Server Distillation・Nested Bottleneck）を検証したが、いずれもベースライン以下に留まった。これはギャップが単一の要因ではなく、「幅スケーリング × アーキテクチャ混在 × Non-IID」という 3 要素の複合的な相互作用に起因することを示唆する。根本的な解決には、HeteroFL の構造制約を保ちながら FedGen の Generator を sub-rate 分布に適応させる新たなアーキテクチャ設計が必要である。

### 6.3 各手法の位置づけと知見

| 手法 | アプローチの方向 | 対象問題 | Phase 2 BEST | 評価 |
|------|---------------|---------|:------------:|:----:|
| **AFAD + Proto** | Generator 側＋クライアント側の二重アンカリング | 潜在空間不整合 | **68.00%** | ✓ AFAD 系最良 |
| **AFAD + RateCond** | Generator を rate 条件付きに拡張 | 潜在空間不整合（直接的） | 67.94% | △ 僅差 |
| **AFAD + AnchorKD** | 凍結フルレートモデルを教師として KD | sub-rate 容量不足 | 66.97% | △ 限定的改善 |
| **AFAD + BNAnchorKD** | AnchorKD + BN 特徴レベル整合 | sub-rate 容量不足 | 66.79% | △ 限定的改善 |
| **AFAD + ServerDistill** | サーバー側クラスバランス蒸留 | Non-IID データ偏り | 67.14% | × Generator 過学習 |
| **AFAD + NestedBN** | 階層的潜在部分空間の共有 | 潜在空間構造化 | 62.53% | × 学習阻害 |
| **AFAD + FedAvg** | FedAvg 集約への切り替え | 集約安定性 | 59.35% | × 構造的不適合 |

Proto と RateCond は「生成側の整合」を、AnchorKD は「識別側の整合」を担う。両者を組み合わせた手法が今後の有望な方向と考えられる。

### 6.4 MNIST IID における新手法の低迷

MNIST IID では RateCond / AnchorKD 系が AFAD Hybrid を下回る（93% 台 vs 99% 台）。これは：

- **タスクの単純性**: MNIST は容量の小さな sub-rate モデルでも十分な精度を出せるため、追加の正則化が過剰になる。
- **IID 環境の特性**: クライアント間のデータ分布が同じため、FedGen の KD が既に十分な情報を提供している。

Non-IID 環境では逆に潜在空間整合が重要になるため、Phase 2 での評価が本質的な判断基準となる。

---

## 7. 各手法の比較表

### 7.1 Phase 2 精度サマリー（OrganAMNIST Non-IID, 40 rounds）

| 手法 | BEST | Final | vs Hybrid |
|------|:----:|:-----:|:---------:|
| HeteroFL Only | 65.36% | 64.90% | −2.48pp |
| AFAD Hybrid | 67.84% | 67.73% | — |
| AFAD + AnchorKD | 66.97% | 66.69% | −0.87pp |
| AFAD + BNAnchorKD | 66.79% | 66.79% | −1.05pp |
| AFAD + ServerDistill | 67.14% | 66.63% | −0.70pp |
| AFAD + RateCond | 67.94% | 67.68% | +0.10pp |
| **AFAD + Proto** | **68.00%** | **67.39%** | **+0.16pp** |
| AFAD + NestedBN | 62.53% | 60.97% | −4.76pp |
| AFAD + FedAvg | 59.35% | 58.53% | −8.20pp |
| FedGen Only | 84.66% | 84.62% | +16.82pp |

### 7.2 手法特性比較

| | HeteroFL Only | FedGen Only | AFAD Hybrid | AFAD + Proto | AFAD + RateCond | AFAD + AnchorKD | AFAD + BNAnchorKD |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 計算能力の異種性（rate 可変） | ○ | × | **○** | **○** | **○** | **○** | **○** |
| アーキテクチャ異種性（CNN ↔ ViT） | × | ○ | **○** | **○** | **○** | **○** | **○** |
| sub-rate KD の有効性 | × | △ | ○ | **◎** | **◎** | **◎** | **◎** |
| Non-IID 耐性 | 中 | **高** | 中 | 中〜高 | 中〜高 | 中〜高 | 中〜高 |
| 集約方式 | count-based | FedAvg | count-based | count-based | count-based | count-based | count-based |
| クライアント損失 | CE | CE + KD (固定 α/β) | CE + KD (rate-dep.) | CE + KD + Proto | CE + KD + Proto | CE + KD + AnchorKD (logit) | CE + KD + AnchorKD (logit+BN) |
| Generator 教師の対象 | なし | rate=1.0 のみ | rate=1.0 のみ | **全 rate** | **全 rate（rate-cond.）** | rate=1.0 のみ | rate=1.0 のみ |
| Prototype Anchoring | × | × | × | ✓ (γ=1.0) | ✓ (γ=1.0) | × | × |
| AnchorKD（凍結教師） | × | × | × | × | × | ✓ (γ=1.0) | ✓ (γ=1.0, BN) |
| サーバー追加コスト | なし | Generator 訓練 | Generator 訓練 | Generator 訓練（6モデル） | Generator 訓練（rate-cond.） | Generator 訓練 + anchor 配布 | Generator 訓練 + anchor 配布 |

---

## 8. システムアーキテクチャ

```
Server（AFADStrategy）
│
├── configure_fit()
│   ├── _initialize_family_models()    — family ごとに rate=1.0 グローバルモデルを初期化
│   ├── HeteroFL distribute            — family_global_models からサブモデルを切り出して配布
│   ├── Cosine LR スケジューラ           — 全クライアントに lr を配信
│   ├── Generator params（pickle）     — warmup 後・AFAD/FedGen のみ
│   └── anchor params（pickle）        — anchor_kd_for_subrate=True かつ rate<1.0 のクライアントのみ
│
├── aggregate_fit()
│   ├── family ごとに結果をグループ化
│   ├── enable_heterofl=True  → _aggregate_heterofl()  — count-based + label-split
│   │   enable_heterofl=False → _aggregate_fedavg()    — サンプル数重み付き平均
│   └── _train_generator_on_server()   — 各 family × rate モデルを再構築して Generator を訓練
│
├── configure_evaluate()
│   └── 各クライアントに対応する family のサブモデルを配信
│
└── aggregate_evaluate()
    └── 加重平均で全体精度を集約

Clients
├── HeteroFLClient  — CE 損失のみ・shape-aware set_parameters・KD なし
├── FedGenClient    — CE + KD（α=β=10 固定）・フルレート・FedAvg 前提
└── AFADClient      — CE + rate-dep. KD（α=β=0.5/rate）・幅スケール・HeteroFL 前提
                       ├── shape-aware set_parameters（幅に応じた部分ロード）
                       ├── FedProx（μ=0.01、Non-IID 安定性）
                       ├── Prototype Anchoring（proto_gamma > 0 時に有効）
                       └── AnchorKD（anchor_kd_gamma / bottleneck_gamma > 0 時に有効）
                             ├── logit-level: T²·KL(f_s(x)/T ‖ f_a(x)/T)
                             └── BN-level: MSE(LN(z_s[:,:ed]), LN(z_a[:,:ed]))
```

---

## 9. 原著論文との差分

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
| Generator 学習教師 | 全クライアントモデル | family ごとの rate=1.0 モデル（AFAD Hybrid） / 全 rate モデル（Proto / RateCond） |
| 対象クライアント | 同一幅が前提 | 幅スケーラブル（rate 混在） |

---

## 10. セットアップ・実行方法

### 動作環境

- Python 3.10+
- PyTorch 2.0+
- Flower 1.7+
- CUDA（推奨、CPU 動作可）

### インストール

```bash
git clone https://github.com/Richiesss/AFAD.git
cd AFAD

# uv のインストール（未導入の場合）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 依存パッケージのインストール
uv sync
```

### 実行

```bash
# --- Phase 1: MNIST IID ---

# 直接シミュレーション（推奨・高速）
uv run python scripts/run_direct_sim.py

# Flower シミュレーション（3 手法比較）
uv run python scripts/run_comparison.py

# 特定手法のみ
uv run python scripts/run_comparison.py \
  --methods "AFAD + Proto,AFAD + RateCond,AFAD + AnchorKD,AFAD + BNAnchorKD"

# --- Phase 2: OrganAMNIST Non-IID ---

uv run python scripts/run_comparison.py config/afad_phase2_config.yaml

# 既存結果をロードして追加実験
uv run python scripts/run_comparison.py config/afad_phase2_config.yaml \
  --methods "AFAD + RateCond" --load results/phase2.json --output results/phase2_new.json

# Multi-seed 統計的検証
uv run python scripts/run_multi_seed.py config/afad_phase2_config.yaml
```

### 開発

```bash
# テスト
uv run pytest -v

# Lint / Format
uv run ruff check . && uv run ruff format --check .
uv run ruff check --fix . && uv run ruff format .  # 自動修正

# 一括実行
uv run poe all
```

---

## 11. ディレクトリ構造

```
AFAD/
├── scripts/
│   ├── run_direct_sim.py           # 直接シミュレーション（推奨・Ray 不要）
│   ├── run_quick_test.py           # クイック検証（4 clients, 5 rounds）
│   ├── run_comparison.py           # 手法比較（Flower + Ray）
│   ├── run_multi_seed.py           # Multi-seed 統計的検証
│   └── run_experiment.py           # 単一実験スクリプト
├── src/
│   ├── client/
│   │   ├── afad_client.py          # AFAD ハイブリッドクライアント（Proto / AnchorKD 対応）
│   │   ├── heterofl_client.py      # HeteroFL Only ベースライン
│   │   └── fedgen_client.py        # FedGen Only ベースライン
│   ├── data/
│   │   ├── dataset_config.py       # データセット設定レジストリ
│   │   ├── mnist_loader.py         # MNIST IID データローダー
│   │   └── medmnist_loader.py      # OrganAMNIST + Dirichlet 分割
│   ├── models/
│   │   ├── registry.py             # モデルレジストリ（ファクトリパターン）
│   │   ├── fedgen_wrapper.py       # FedGenModelWrapper（bottleneck + classifier）
│   │   ├── scaler.py               # HeteroFL Scaler（1/rate 補償）
│   │   ├── cnn/
│   │   │   ├── heterofl_resnet.py  # 幅スケーラブル ResNet18（sBN + Scaler）
│   │   │   ├── resnet.py           # ResNet18, ResNet50
│   │   │   └── mobilenet.py        # MobileNetV3-Large
│   │   └── vit/
│   │       ├── heterofl_vit.py     # 幅スケーラブル ViT-Small（sBN + Scaler）
│   │       ├── vit.py              # ViT-Tiny, ViT-Small
│   │       └── deit.py             # DeiT-Small
│   ├── server/
│   │   ├── generator/
│   │   │   ├── fedgen_generator.py         # FedGen 潜在空間 Generator（rate-cond. 対応）
│   │   │   └── afad_generator_trainer.py   # サーバーサイド Generator 訓練（rate-aware）
│   │   └── strategy/
│   │       ├── afad_strategy.py            # AFAD 統合戦略（全モード対応）
│   │       └── heterofl_aggregator.py      # HeteroFL 集約（count-based + label-split）
│   └── utils/
│       ├── config_loader.py        # YAML 設定読み込み
│       ├── logger.py               # ロガー
│       └── metrics.py              # MetricsCollector
├── tests/
│   ├── test_afad_integration.py    # AFAD E2E 統合テスト
│   ├── test_heterofl_aggregator.py # HeteroFL 集約テスト
│   ├── test_fedgen_faithful.py     # FedGen コンポーネントテスト
│   └── test_generator.py           # Generator テスト
├── config/
│   ├── afad_config.yaml            # Phase 1 設定（MNIST, IID）
│   └── afad_phase2_config.yaml     # Phase 2 設定（OrganAMNIST, Non-IID）
├── results/                        # 実験結果 JSON
├── pyproject.toml
└── uv.lock
```

---

## 12. 参考文献

- Diao, E., Ding, J., and Tarokh, V. "HeteroFL: Computation and Communication Efficient Federated Learning for Heterogeneous Clients." *International Conference on Learning Representations (ICLR)*, 2021.
- Zhu, Z., Hong, J., and Zhou, J. "Data-Free Knowledge Distillation for Heterogeneous Federated Learning." *International Conference on Machine Learning (ICML)*, 2021.
- Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Smola, A., and Smith, V. "Federated Optimization in Heterogeneous Networks." *Proceedings of Machine Learning and Systems (MLSys)*, 2020.
- Hinton, G., Vinyals, O., and Dean, J. "Distilling the Knowledge in a Neural Network." *NeurIPS Deep Learning Workshop*, 2015.
- Horvath, S., Laskaridis, S., Almeida, M., Leontiadis, I., Venieris, S., and Lane, N. "FjORD: Fair and Accurate Federated Learning under heterogeneous targets with Ordered Dropout." *Neural Information Processing Systems (NeurIPS)*, 2021.

---

## 著者

- **作成者**: 島野 凌
- **所属**: 大阪工業大学 大学院 情報科学研究科
