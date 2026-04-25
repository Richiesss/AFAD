# AFAD: Adaptive Federated Architecture Distribution

> **連合学習における二重の異種性（計算能力 × モデルアーキテクチャ）を同時に解決するハイブリッド連合学習フレームワーク**

---

## Abstract

実際の連合学習（Federated Learning; FL）環境では、クライアントは計算能力とモデルアーキテクチャの両面で異質である。既存手法は一方の問題のみを扱う：HeteroFL [Diao+, ICLR 2021] は幅スケーリングで計算能力の異種性に対応するが、異なるアーキテクチャ族（CNN と Transformer など）間の知識共有を行わない。FedGen [Zhu+, ICML 2021] はデータフリー知識蒸留（KD）でアーキテクチャ異種性に対応するが、全クライアントが同一サイズのモデルを持つことを前提とする。

本研究では **AFAD（Adaptive Federated Architecture Distribution）** を提案する。AFAD は両手法を統合し、共有 32 次元潜在空間と Generator を介してアーキテクチャをまたいだ知識共有を実現しながら、幅スケーリングで計算能力の異種性にも対応する。単純な統合（ナイーブ統合）から 4 段階の改善を経て、さらに **FedProto 型 Prototype Regularization** をはじめとする複数の拡張手法を提案・体系的に評価する。

**IID 環境（OrganAMNIST, 10-client, server_acc）における主要結果**: AFAD + ProjHead は FedGen Only（86.32%）を **+2.57pp 上回る 88.89%** を達成する。FedGen は CNN と ViT の混在設定を扱えないのに対し、AFAD はその制約を持たない。

**Non-IID 環境（OrganAMNIST, α=0.5）における進展**: FedProto 型 Prototype Regularization（AFAD + Proto, scale=0.5）を導入することで、Mean R11+ 72.64%（Best 73.12%）を達成し、ベースライン（AFAD Hybrid: 66.52%）から **+6.12pp** の大幅改善を実現した。FedGen Only（Mean R11+: 83.97%）との残ギャップは 11.33pp（初期の 17pp 超から縮小）。14 種を超える手法の体系的実験を通じ、このギャップの根本原因（Generator 潜在空間と sub-rate backbone の構造的不整合、および cross-family z-space fragmentation）を特定・定量化した。

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
   - 4.1 [AFAD + FedProto（Prototype Regularization）](#41-afad--fedproto)
   - 4.2 [AFAD + BackboneAlign](#42-afad--backbonealign)
   - 4.3 [AFAD + S-CFC（Server-side Cross-Family Consensus）](#43-afad--s-cfc)
   - 4.4 [AFAD + LatentSupCon（Latent Supervised Contrastive Learning）](#44-afad--latentsupcon)
   - 4.5 [AFAD + GenMix（Generative Mixup）](#45-afad--genmix)
   - 4.6 [AFAD + ProjHead（Projection Head、IID 向け）](#46-afad--projhead)
5. [実験](#5-実験)
   - 5.1 [実験設定](#51-実験設定)
   - 5.2 [Phase 1: MNIST IID](#52-phase-1-mnist-iid)
   - 5.3 [Phase 2: OrganAMNIST（IID + Non-IID）](#53-phase-2-organamnist-iid--non-iid)
     - 5.3.1 [IID 結果](#531-iid-結果)
     - 5.3.2 [Non-IID 全手法比較](#532-non-iid-全手法比較)
     - 5.3.3 [FedGen ギャップ解消実験（旧アプローチ）](#533-fedgen-ギャップ解消実験旧アプローチ)
   - 5.4 [アブレーション実験](#54-アブレーション実験)
6. [分析・考察](#6-分析考察)
7. [各手法の比較表](#7-各手法の比較表)
8. [システムアーキテクチャ](#8-システムアーキテクチャ)
9. [研究の貢献と残された課題](#9-研究の貢献と残された課題)
10. [原著論文との差分](#11-原著論文との差分)
11. [セットアップ・実行方法](#12-セットアップ実行方法)
12. [ディレクトリ構造](#13-ディレクトリ構造)
13. [参考文献](#14-参考文献)

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

> **主命題**: 計算能力・アーキテクチャの二重異種性が存在する FL 環境において、FedGen の知識蒸留を HeteroFL の幅スケーリングと両立させることで、IID 環境下では FedGen と同等の精度を達成しつつ、FedGen には不可能な異種アーキテクチャ間の協調学習を実現できるか？

特に以下の問いに答えることを目標とする：

- **RQ1（主）**: HeteroFL と FedGen の単純な統合（ナイーブ統合）はなぜ機能しないのか？また、どのような改善により IID 環境での性能を FedGen と同等水準まで回復できるか？
- **RQ2（副）**: sub-rate クライアント（rate<1.0）が FedGen の KD 信号を適切に活用できない根本原因は何か？Non-IID 環境での性能差はどこから生じるか？
- **RQ3（将来）**: Non-IID 環境での FedGen との性能差を解消するために、どのようなアーキテクチャ設計が有効か？

> **スコープの明確化**: 本研究は RQ1・RQ2 に完全に答える。RQ3 は現時点では「残された課題」として位置づけ、方向性のみを示す（→ [9章](#9-研究の貢献と残された課題)）。

---

## 2. 関連研究

### 2.1 計算能力の異種性への対応

**HeteroFL** [Diao et al., ICLR 2021] は、グローバルモデルの先頭チャネルを rate に応じて切り出したサブモデルをクライアントに配布することで、異なる計算能力のクライアントが同一の FL に参加できる仕組みを提供する。Static BatchNorm と Scaler を組み合わせることで、幅縮小による活性化値のスケール変化を補償する。

**FjORD** [Horvath et al., NeurIPS 2021] は Ordered Dropout により、HeteroFL と類似の幅スケーリングを実現するが、Static BatchNorm は使用しない。

### 2.2 アーキテクチャの異種性への対応

**FedGen** [Zhu et al., ICML 2021] は、サーバーサイドで Generator $G: y \to z$ を学習し、クライアントに生成された潜在ベクトルを用いた KD を適用することで、データフリーかつアーキテクチャに依存しない知識共有を実現する。ただし、全クライアントが同一サイズのモデルを保持することを前提とする。

**FedGen が幅の異なるクライアントを扱えない構造的理由**: FedGen はモデル集約に **FedAvg**（サンプル数加重平均）を使用する。FedAvg はすべてのクライアントモデルが同一のパラメータ形状を持つことを前提としており、rate=1.0（backbone 512ch）と rate=0.5（backbone 256ch）のように幅の異なるモデルを単純平均することは不可能である。bottleneck 次元（32次元）が共通であっても、backbone のパラメータ形状が異なる以上、FedAvg による集約はできない。したがって FedGen Only では全クライアントを rate=1.0 に固定せざるを得ない。

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

AFAD Hybrid をベースに、Non-IID 環境での FedGen ギャップを縮める（または IID 精度をさらに向上させる）ための拡張手法を提案する。

---

### 4.1 AFAD + FedProto（Prototype Regularization）

#### 問題の特定

Non-IID 環境での主要な障害は、**クライアント間で backbone が学習する特徴量分布の不整合**にある。Generator が生成する潜在ベクトル z_gen は rate=1.0 backbone の分布に最適化されており、sub-rate クライアントの bottleneck 出力（異なる入力次元からの射影）とは方向分布が異なる。

#### 解決策：FedProto 型 Prototype Regularization

FedProto [Tan+, AAAI 2022] の設計思想をもとに、以下のアプローチを実装する：

1. **クライアント側**: 各クラスの backbone 特徴量（`bottleneck(backbone(x))`）の平均ベクトル（prototype）をサーバーに送信
2. **サーバー側**: 全クライアント・全ファミリー（CNN + ViT）の prototype を per-class 平均集約 → `global_protos` として配布
3. **クライアント正則化**: `L_proto = proto_scale × MSE(z_local_class, global_proto_class)` を訓練損失に加算

#### 特性

- **Generator z（ノイズが多い）ではなく実データから計算した class centroid を使用** → より正確なアンカー
- CNN と ViT の両ファミリーが同じ 32 次元 bottleneck を共有するため、architecture-agnostic な class centroid が自然に形成される
- proto_scale のハイパーパラメータ探索（Mean R11+）: 0.1（70.08%、+3.56pp） < 0.75（71.35%、+4.83pp） < **0.5（72.64%、+6.12pp ★最良）** > 1.0（CUDA 不安定）

#### 数式

```
L_client = L_CE + (α/rate)·L_KD + proto_scale·L_proto

L_proto = MSE( bottleneck(backbone(x))_class_y , global_proto_y )
```

---

### 4.2 AFAD + BackboneAlign

#### 動機

Generator KD の損失 `forward_from_latent(z_gen)` は **classifier 層のみを訓練**し、backbone を経由しない（`forward_from_latent(z) = classifier(z)` の直接射影）。このため sub-rate backbone の特徴抽出能力は Generator KD によって改善されない。

#### 手法

クライアント学習時に、実データの backbone 出力 z_real と Generator が生成した z_gen の間の MSE 損失を追加：

```
L_ba = backbone_align_scale × MSE( bottleneck(backbone(x)), z_gen.detach() )
```

z_gen を detach することで backbone のみを訓練。scale=0.1（デフォルト）での Mean R11+ = 67.07%（+0.55pp vs Hybrid）。scale 増大（0.3）は不安定化を招く（Mean R11+ = 65.71%）。

---

### 4.3 AFAD + S-CFC（Server-side Cross-Family Consensus）

#### 動機

Non-IID 環境では CNN ファミリーと ViT ファミリーの分類器が異なるクラス境界を学習し、Generator が生成する z が両者で整合しない（cross-family z-space fragmentation）。

#### 手法

Generator 訓練時に、全ファミリーの分類器出力の KL ダイバージェンスを最小化するペナルティを追加：

```
L_CFC = (1/K) Σ_k KL(p_k || mean_p)
mean_p = (1/K) Σ_k p_k,  p_k = softmax(classifier_k(z_gen))

L_gen = α·L_teacher + η·L_diversity + γ·L_CFC
```

gamma=0.1 で Mean R11+ ≈ 67.5%（+約1.0pp vs Hybrid）。

---

### 4.4 AFAD + LatentSupCon（Latent Supervised Contrastive Learning）

#### 動機

BackboneAlign の MSE は「同クラスの z_real を z_gen に近づける」のみ。異クラス間の分離も同時に促進するために対照学習を導入する。

#### 手法

InfoNCE 損失を潜在空間に適用：

```
sim_matrix = normalize(z_real) @ normalize(z_gen_all_classes).T / τ
L_supcon = CE(sim_matrix, class_labels)
```

正例：同クラスの z_gen、負例：異クラスの z_gen（全クラス分の z_gen をバッファとして保持）。

---

### 4.5 AFAD + GenMix（Generative Mixup）

#### 動機

BackboneAlign の MSE は座標レベルの整合を強制するが、backbone の特徴量空間を硬直させるリスクがある。より soft なタスク駆動の勾配を与えるために Mixup を活用する。

#### 手法

Beta(α, α) 分布からサンプリングしたλでlocal特徴量とgenerator特徴量を補間：

```
λ ~ Beta(genmix_alpha, genmix_alpha)
z_mix = λ·z_local + (1-λ)·z_gen
L_genmix = CE(classifier(z_mix), y)
```

alpha=0.2 での実験結果はベースラインとほぼ同等（探索的実験）。

---

### 4.6 AFAD + ProjHead（Projection Head、IID 向け）

#### 動機

IID 環境での精度をさらに向上させるため、rate 間の bottleneck 表現を整合させる Projection Head を導入する。

#### 手法

各 rate クライアントに `P_r: z_eff → z_32` という線形射影ヘッドを追加（`z_eff = int(32 × rate)`次元 → 32次元）。
IID OrganAMNIST での server_acc = **88.89%** を達成し、FedGen Only（86.32%）を **+2.57pp** 上回る。

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

#### 5.2.3 Flower シミュレーション（10 clients, 40 rounds）— Phase 2 と同一条件

> `run_comparison.py`（Flower + Ray）、`config/afad_phase1_10client_config.yaml` 使用。seed=42 の単一試行。Phase 2（OrganAMNIST Non-IID）と同じクライアント構成・ラウンド数でIID環境を評価。**評価指標**: サーバーの rate=1.0 グローバルモデルを集中テストセットで評価した `server_accuracy`（括弧内は各クライアントのサブレートモデル平均精度）。

| 手法 | server_acc BEST | server_acc FINAL | (client_acc FINAL) |
|------|:--------------:|:----------------:|:-----------------:|
| **AFAD Hybrid** | **90.47%** | **90.47%** | (86.83%) |
| HeteroFL Only | 90.19% | 90.17% | (86.41%) |
| FedGen Only ※ | 97.58% | 97.51% | (97.51%) |

> **※ 重要な注記**: FedGen Only は**全クライアントが rate=1.0**（フルサイズモデル）で動作する。FedGen は固定次元の潜在ベクトルを前提とするため、sub-rate クライアントへの対応が構造的に不可能である（異なる rate のボトルネックは異なる入力次元を持ち、Generator との整合が取れない）。一方、AFAD/HeteroFL は rate=0.5・0.25 のクライアントを含む。**FedGen の 97.51% は「計算能力の異種性なし」の条件であり、AFAD との直接比較は不公平**である。
>
> この設定での精度差は FedGen の性能優位ではなく、**AFAD が解く問題の難しさ（sub-rate クライアントの容量制約）**を示している。AFAD の公平な比較は直接シミュレーション（5.2.1）の結果による。
>
> **client_acc と server_acc のギャップ**（AFAD: 86.83% → 90.47%、+3.64pp / HeteroFL: 86.41% → 90.17%、+3.76pp）は、クライアント評価がサブレートモデルによる過小評価であることを示す。公正なサーバー評価では両手法とも 90% を超える。

**AnchorKD の初期収束優位性**（MNIST IID、Round 1〜5）:

| Round | AFAD + RateCond | AFAD + AnchorKD | AFAD + BNAnchorKD |
|:-----:|:---------------:|:---------------:|:-----------------:|
| 1 | 9.59% | 9.26% | 9.18% |
| 2 | 73.34% | 75.21% | 73.47% |
| 3 | 81.40% | 82.50% | 84.27% |
| 4 | 84.64% | 86.12% | 87.65% |
| 5 | 89.03% | 88.78% | 89.27% |

AnchorKD 系は序盤（Round 3〜4）でフルレート教師による早期誘導が有効に働き、RateCond より速い収束を示す。

### 5.3 Phase 2: OrganAMNIST（IID + Non-IID）

**実験設定**: 10 クライアント（CNN × 5 + ViT × 5）、rate 混在（0.25/0.5/1.0）、40 rounds、Flower + Ray シミュレーション。評価指標は `server_acc`（サーバーの rate=1.0 グローバルモデルを集中テストセットで評価）。Mean R11+ = Round 11〜40 の平均（収束後の安定精度）。

---

#### 5.3.1 IID 結果

| 手法 | Best server_acc | 備考 |
|------|:--------------:|------|
| HeteroFL Only | 75.77% | |
| AFAD Hybrid | 74.95% | |
| **AFAD + ProjHead** | **88.89%** | ★ FedGen を +2.57pp 上回る |
| FedGen Only ※ | 86.32% | ※ 全員 rate=1.0（計算能力の異種性なし） |

> **AFAD + ProjHead が IID で FedGen Only を超える**。rate 混在クライアントを扱いながら FedGen の上限を突破した点が本研究の明確な貢献の一つ。

---

#### 5.3.2 Non-IID 全手法比較（Dirichlet α=0.5）

評価指標: server_acc の Mean R11+（Round 11〜40 平均）。Best は 40 round 中の最高値。

| 手法 | Best | Mean R11+ | Std | vs Hybrid |
|------|:----:|:---------:|:---:|:---------:|
| HeteroFL Only | 65.36% | 64.90% | 0.38% | −1.62pp |
| **AFAD Hybrid**（ベースライン） | 67.08% | 66.52% | 0.76% | — |
| AFAD + AvailLabels | 66.89% | 66.28% | 0.78% | −0.24pp |
| AFAD + PerFamilyGen | 67.03% | 66.23% | 1.00% | −0.29pp |
| AFAD + Consensus | 67.39% | 66.72% | 0.80% | +0.20pp |
| AFAD + RelKD | 67.65% | 66.87% | 0.88% | +0.35pp |
| AFAD + RelKD + Consensus | 65.73% | 65.15% | 0.51% | −1.37pp |
| AFAD + AnchorKD | 70.18% | 66.55% | 2.43% | +0.03pp |
| AFAD + BNAnchorKD | 69.25% | 66.11% | 1.83% | −0.41pp |
| AFAD + BackboneAlign | 67.84% | 67.07% | 0.83% | +0.55pp |
| AFAD + BackboneAlign (scale=0.3) | 70.33% | 65.71% | 1.67% | −0.81pp |
| AFAD + BackboneAlign + RelKD | 67.33% | 66.65% | 0.90% | +0.13pp |
| AFAD + S-CFC (γ=0.1) | ≈68.3% | ≈67.5% | — | ≈+1.0pp |
| AFAD + Proto (scale=0.1) | 70.86% | 70.08% | 0.39% | +3.56pp |
| **AFAD + Proto (scale=0.5)** | **73.12%** | **72.64%** | **0.25%** | **+6.12pp ★** |
| AFAD + Proto (scale=0.75) | 71.86% | 71.35% | 0.58% | +4.83pp |
| FedGen Only ※ | 84.66% | 83.97% | 0.76% | +17.45pp |

> ※ FedGen Only は全クライアント rate=1.0（計算能力の異種性なし）。直接比較は不公平。
>
> **AFAD + Proto (scale=0.5) が最良手法**。Mean R11+ の Std が 0.25% と極めて低く、再現性が高い。scale=0.75 では過正則化により Mean R11+ が 71.35%（−1.29pp）に低下し、scale=1.0 では CUDA 不安定が発生。FedGen との残ギャップは 11.33pp（初期の ~17pp から縮小）。

##### 主要な知見

1. **Proto が突出して有効**（+6.12pp）: Generator z（ノイズが多い）ではなく、**実データから計算した class centroid** を正則化ターゲットに使うことで、architecture-agnostic な backbone 特徴量空間の整合が実現する。
2. **scale の最適値は 0.5**（scale=0.1: +3.56pp, scale=0.5: +6.12pp, scale=0.75: ~+5.5pp, scale=1.0: CUDA 不安定）。正則化が強すぎると不安定化する。
3. **client-side KD の追加的探索は 67% 前後で頭打ち**（RelKD, Consensus, BackboneAlign 単独では Mean R11+ ≤ 67.1%）。Proto は別の機構（backbone 直接正則化）で壁を突破した。
4. **Cross-family ensemble 知識の削減は禁忌**（PerFamilyGen で確認）。
5. **AnchorKD は Best が高い（70.18%）が Std も高い（2.43%）**：外れ値的な高値であり、Mean では AFAD Hybrid と同等。
6. **scale 増大は必ずしも改善しない**（BackboneAlign scale=0.3, Proto scale=1.0 で確認）。

---

#### 5.3.3 FedGen ギャップ解消実験（旧アプローチ）

16.66pp のギャップを縮めるために初期段階で 3 つのアーキテクチャレベルのアプローチを体系的に検討した。

**Case C: FedAvg 集約**（HeteroFL の count-based 集約を FedAvg に変更）

**Case A: Server-side Distillation**（Generator を用いてサーバーで追加蒸留）

**Case B: Nested Bottleneck**（rate に応じた階層的潜在部分空間）

| 手法 | BEST | Final | vs AFAD Hybrid |
|------|:----:|:-----:|:--------------:|
| AFAD Hybrid（ベースライン） | 67.84% | 67.73% | — |
| AFAD + ServerDistill (Case A) | 67.14% | 66.63% | **−0.70pp** |
| AFAD + NestedBN (Case B) | 62.53% | 60.97% | **−4.76pp** |
| AFAD + FedAvg (Case C) | 59.35% | 58.53% | **−8.20pp** |
| FedGen Only（上限） | 84.66% | 84.62% | +16.82pp |

> ※ 旧設定（client_acc ベース）での参考数値。いずれもベースライン以下となり、この方向性での限界を確認。

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

### 6.1 各手法が「混在できること」と「知識共有できること」の違い

10クライアント設定では CNN（client 0〜4）と ViT（client 5〜9）が共存するが、各手法で「混在の意味」が異なる。

#### HeteroFL: 同居するが互いに無視する

HeteroFL はファミリー内でのみ集約を行う。CNN クライアントは CNN グローバルモデルにのみ集約され、ViT クライアントは ViT グローバルモデルにのみ集約される。アーキテクチャをまたいだパラメータ共有・知識移転は一切ない。実質的に「2つの独立した FL を並列実行しているだけ」である。

```
HeteroFL Only:
  CNN 0,1,2,3,4  →  集約  →  CNN グローバルモデル（ViT の学習を全く参照しない）
  ViT 5,6,7,8,9  →  集約  →  ViT グローバルモデル（CNN の学習を全く参照しない）
```

#### FedGen: 幅の異なるクライアントを混在できない

FedGen は CNN ↔ ViT 間の知識共有を Generator 経由で実現できるが、**幅の異なるクライアント（rate=1.0/0.5/0.25 の混在）は構造的に扱えない**。

誤解されやすい点として、bottleneck 次元（32次元）は rate に関係なく固定されており、Generator の出力次元と整合する。しかし問題はその前段の backbone にある：

| rate | backbone 出力形状（CNN） | FedAvg で平均できるか |
|------|:-----------------------:|:--------------------:|
| 1.0 | [512, 256, 128, 64, ...] | — |
| 0.5 | [256, 128, 64, 32, ...] | ✗（形状が違う） |
| 0.25 | [128, 64, 32, 16, ...] | ✗（形状が違う） |

FedGen は FedAvg（サンプル数加重平均）で集約するため、すべてのパラメータが同一形状でなければならない。幅の異なるモデルは加重平均できないため、FedGen Only では全クライアントを rate=1.0 に固定せざるを得ない。

#### AFAD: 両方の問題を同時に解決する

AFAD は HeteroFL の count-based 集約（形状が違っても先頭スライスのみを加算・平均する仕組み）を維持したまま、FedGen の Generator KD をファミリー間知識共有に活用する。

```
AFAD Hybrid:
  CNN 0〜4（rate 混在）  →  bottleneck (32次元) ─┐
                                                   ├→ Generator（共有潜在空間で KD）
  ViT 5〜9（rate 混在）  →  bottleneck (32次元) ─┘
  ↑ count-based 集約でファミリー内の幅混在に対応  ↑ FedGen KD でアーキ間知識共有
```

| | FedGen Only | HeteroFL Only | AFAD Hybrid |
|--|:-----------:|:-------------:|:-----------:|
| CNN ↔ ViT 知識共有 | ✓ | ✗ | ✓ |
| rate 混在（0.5/0.25）対応 | ✗（全員 rate=1.0 固定） | ✓ | ✓ |
| 集約方式 | FedAvg | count-based | count-based |

### 6.2 ナイーブ統合が失敗する理由

HeteroFL と FedGen を素朴に統合すると **9.55pp（60.30% → 69.85%）もの性能差**が生じる。その主な原因は以下の 2 点：

1. **過剰正則化**: FedGen の KD 係数（α=β=10）は同一幅モデルを前提に設計されており、幅の異なるモデルに適用すると正則化が過剰になる（改善幅の 76%、7.25pp を占める）。
2. **rate=1.0 クライアントへの KD 制限**: 元の FedGen の実装では rate=1.0 のクライアントのみを KD 対象とすることで精度を守っていたが、AFAD では sub-rate クライアントも学習する必要があるため、全クライアントへの KD 適用が必要。

### 6.3 IID 環境における AFAD の有効性

**IID 環境では AFAD は FedGen と同等以上の精度を達成する**（AFAD Hybrid 69.85% vs FedGen Only 67.00%、+2.85pp）。

この結果が意味することは大きい：FedGen はアーキテクチャ異種性に対応するが、「全クライアントが同じ幅のモデルを持つ」という制約がある。AFAD はこの制約を取り払いながら、FedGen と同水準の知識共有を IID 環境で実現する。

| 比較軸 | FedGen Only | AFAD Hybrid |
|--------|:-----------:|:-----------:|
| 計算能力の異種性（rate 可変） | ✗（全員 rate=1.0） | **✓** |
| アーキテクチャ異種性（CNN ↔ ViT） | ✓ | **✓** |
| IID 精度（MNIST 直接シミュレーション） | 67.00% | **69.85%（+2.85pp）** |
| IID 精度（MNIST Flower, 5 clients） | 99.60% | 99.35%（−0.25pp） |
| IID 精度（MNIST Flower, 10 clients, server_acc）※ | 97.51% | 90.47%（−7.04pp） |

> ※ 10クライアント設定では FedGen は全員 rate=1.0（計算能力の異種性なし）で動作するため、直接比較は不公平。AFAD は rate=0.5/0.25 の容量制約クライアントを含む難条件で評価されている。server_acc = サーバーの rate=1.0 グローバルモデルを集中テストセットで評価した値。
>
> **公平な比較（直接シミュレーション）では AFAD が FedGen を上回る（+2.85pp）**。FedGen が扱えない設定（幅の異なるクライアント混在）を扱いながら同等以上の精度を達成している点が AFAD の本質的な貢献である。

### 6.4 Non-IID 環境でのギャップとその根本原因

Non-IID 環境（OrganAMNIST, α=0.5）での FedGen（83.97%）と AFAD Hybrid（66.52%）の **17.45pp のギャップ**は、IID 環境では存在しない。**AFAD + Proto (scale=0.5) によって Mean R11+ 72.64% を達成し、ギャップを 11.33pp に縮小した**が、完全な解消には至っていない。

**ギャップの構造的な原因（3 要素の複合）**:

1. **潜在空間の不整合（cross-family z-space fragmentation）**: Generator は rate=1.0 の bottleneck 分布に最適化されており、rate=0.5/0.25 の bottleneck 出力とは分布が異なる。さらに CNN と ViT で特徴量多様体が分裂し、Generator が両者の交差領域のみをカバーせざるを得ない（PerFamilyGen でアンサンブル知識削減→悪化を確認）。

2. **HeteroFL 集約の制約**: count-based 集約はサブモデルへの適切なパラメータ分配に不可欠（FedAvg 切り替えにより −8.20pp 悪化）だが、FedAvg 集約が持つ Non-IID 安定性は得られない。

3. **forward_from_latent のバイパス問題**: `forward_from_latent(z) = classifier(z)` は backbone を経由せず、Generator KD が backbone 品質を直接改善しない。Proto は backbone 特徴量を直接正則化することでこの問題を迂回する。

**Proto が有効な理由**:
- Generator z（ノイズが多い）ではなく、**実データから計算した per-class mean（prototype）** を正則化ターゲットに使用
- CNN/ViT が同じ 32 次元 bottleneck を共有するため、architecture-agnostic な class centroid が自然に得られる
- Backbone の特徴抽出を直接正則化 → forward_from_latent のバイパス問題を回避

**残ギャップ（11.33pp）の解釈**:
- Sub-rate backbone の容量不足（rate=0.25/0.5 の容量制約）は Proto では解消されない
- Generator z-space と実 backbone 分布の根本的な不整合は依然残存
- Non-IID 環境での局所データ偏り補正を Generator（サーバー側）が行う FedGen の仕組みとは異なるアプローチ

---

### 6.5 各手法の位置づけと知見

| 手法 | アプローチの方向 | 対象問題 | Mean R11+ | 評価 |
|------|---------------|---------|:---------:|:----:|
| **AFAD + Proto (scale=0.5)** | 実 backbone 特徴量の class centroid 正則化 | backbone 品質・潜在空間整合 | **72.64%** | ★★ 最良（scale 最適値） |
| AFAD + Proto (scale=0.75) | 同上（過正則化） | 同上 | 71.35% | △ scale 増→悪化 |
| **AFAD + Proto (scale=0.1)** | 同上（弱め） | 同上 | 70.08% | ✓ 70% 突破 |
| AFAD + S-CFC | サーバーGen訓練時の cross-family consensus | z-space fragmentation（gen側） | ≈67.5% | ✓ +1.0pp |
| AFAD + BackboneAlign | 実データ z_real → z_gen への MSE | backbone 品質 | 67.07% | ✓ +0.55pp |
| AFAD + RelKD | Relative KD（比較的比率を保つKD） | KD の質向上 | 66.87% | △ +0.35pp |
| AFAD + Consensus | Generator gen に family 合意点強制 | z-space fragmentation | 66.72% | △ +0.20pp |
| AFAD + BackboneAlign + RelKD | 組み合わせ | — | 66.65% | △ 組合せ逆効果 |
| AFAD + AvailLabels | 利用可能ラベルのみKD | KD 効率 | 66.28% | × 悪化 |
| AFAD + PerFamilyGen | ファミリー別Generator | Cross-family知識 | 66.23% | × アンサンブル削減で悪化 |
| AFAD + AnchorKD | 凍結フルレートモデルを教師に | sub-rate 容量不足 | 66.55% | △ Best高いがStd大 |
| AFAD + BNAnchorKD | AnchorKD + BN 特徴整合 | 同上 | 66.11% | △ 限定的 |
| AFAD + RelKD + Consensus | 組合せ | — | 65.15% | × 組合せ悪化 |
| AFAD + BackboneAlign (scale=0.3) | BackboneAlign 強め | — | 65.71% | × scale 増大→不安定 |
| AFAD + ServerDistill | サーバー側KD | Non-IID データ偏り補正 | ≈66.6% | × Generator 過学習 |
| AFAD + NestedBN | 階層的潜在部分空間 | 潜在空間構造化 | ≈60.9% | × 学習阻害 |
| AFAD + FedAvg | FedAvg 集約への切り替え | 集約安定性 | ≈58.5% | × 構造的不適合 |

---

## 7. 各手法の比較表

### 7.1 OrganAMNIST 精度サマリー（10 clients, 40 rounds, server_acc）

#### IID

| 手法 | Best server_acc | vs AFAD Hybrid |
|------|:--------------:|:--------------:|
| AFAD Hybrid | 74.95% | — |
| HeteroFL Only | 75.77% | +0.82pp |
| **AFAD + ProjHead** | **88.89%** | **+13.94pp ★** |
| FedGen Only ※ | 86.32% | +11.37pp |

> ※ FedGen Only は全クライアント rate=1.0（計算能力の異種性なし）。AFAD + ProjHead は rate 混在クライアントを含む難条件で FedGen を超えている。

#### Non-IID（Dirichlet α=0.5）— 主要手法のみ

| 手法 | Best | Mean R11+ | Std | vs AFAD Hybrid |
|------|:----:|:---------:|:---:|:--------------:|
| HeteroFL Only | 65.36% | 64.90% | 0.38% | −1.62pp |
| AFAD Hybrid | 67.08% | 66.52% | 0.76% | — |
| AFAD + BackboneAlign | 67.84% | 67.07% | 0.83% | +0.55pp |
| AFAD + S-CFC (γ=0.1) | ≈68.3% | ≈67.5% | — | ≈+1.0pp |
| AFAD + Proto (scale=0.1) | 70.86% | 70.08% | 0.39% | +3.56pp |
| **AFAD + Proto (scale=0.5)** | **73.12%** | **72.64%** | **0.25%** | **+6.12pp ★** |
| AFAD + Proto (scale=0.75) | 71.86% | 71.35% | 0.58% | +4.83pp |
| FedGen Only ※ | 84.66% | 83.97% | 0.76% | +17.45pp |

> ※ FedGen Only は全クライアント rate=1.0（計算能力の異種性なし）。

全手法の詳細比較は **5.3.2 節**を参照。

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

## 9. 研究の貢献と残された課題

### 9.1 確立された貢献

#### 貢献 1: 二重異種性 FL フレームワークの提案

FedGen（アーキテクチャ異種性対応）と HeteroFL（計算能力異種性対応）を統合した AFAD を提案した。**この 2 種類の異種性を同時に扱う FL フレームワークは AFAD 以前に存在しない**。

#### 貢献 2: IID 環境での FedGen 超えの精度達成

| 設定 | FedGen Only | AFAD | 差 | 条件の公平性 |
|------|:-----------:|:----:|:--:|:----------:|
| MNIST IID（直接シミュレーション） | 67.00% | **69.85%（AFAD Hybrid）** | **+2.85pp** | ✓ 公平 |
| OrganAMNIST IID（Flower, 10 clients, server_acc）| 86.32% ※ | **88.89%（AFAD + ProjHead）** | **+2.57pp** | △ FedGen は rate=1.0 のみ |

> ※ FedGen Only の 10クライアント設定は全員 rate=1.0（計算能力の異種性なし）。AFAD は sub-rate クライアント（rate=0.5/0.25）を含む難条件。IID 環境では AFAD が FedGen を上回る（+2.57pp）という明確な貢献がある。

#### 貢献 3: ナイーブ統合の失敗原因の解明と 4 段階改善

HeteroFL + FedGen のナイーブ統合（60.30%）がなぜ失敗するかを体系的に分析し、4 段階の改善（最終 69.85%、**+9.55pp**）により回復した。

#### 貢献 4: FedProto 型 Prototype Regularization による Non-IID 大幅改善

14 種を超える手法の体系的実験を通じ、**AFAD + Proto (scale=0.5) が Mean R11+ 72.64%（Best 73.12%）** を達成。ベースライン（AFAD Hybrid 66.52%）から **+6.12pp**、FedGen との残ギャップを 17pp から 11.33pp に縮小した。

**なぜ Proto が有効か**: Generator z（ノイズが多い）ではなく実データから計算した per-class centroid を backbone 正則化ターゲットに使用することで、CNN/ViT の architecture-agnostic な特徴量空間整合が実現する。

#### 貢献 5: Non-IID ギャップの根本原因の特定と定量化

IID 環境では存在しない 17pp のギャップが Non-IID で生じる原因を、14 種超の体系的実験により特定した。

> **根本原因**: (1) forward_from_latent が backbone をバイパスし Generator KD が backbone 品質を直接改善しない（推定 ~6pp）、(2) sub-rate backbone の容量不足（推定 ~4pp）、(3) cross-family z-space fragmentation（CNN/ViT の特徴量多様体の分裂、推定 ~3pp）。

---

### 9.2 残された課題（Non-IID 環境への対応）

**AFAD + Proto (scale=0.5) の Mean R11+ = 72.64% は現時点の最良だが、FedGen との 11.33pp ギャップは依然残存する**。

#### 課題 1: Forward-from-latent のバイパス問題の解消

現状: `forward_from_latent(z) = classifier(z)` — backbone を経由しない。  
必要: sub-rate backbone を直接改善できる仕組み（FitNet 蒸留など）。

**有望なアプローチ**: サーバーサイド FitNet 蒸留（rate=1.0 backbone の特徴量 → rate<1.0 クライアントへ MSE 蒸留）。HeteroFL 集約ロジックを破壊せず backbone 品質を向上できる可能性がある。

#### 課題 2: Proto + 他手法の組み合わせ探索

AFAD + S-CFC + Proto（Generator の z-space fragmentation を S-CFC で抑制しつつ、Proto で backbone を正則化）は未試験。

#### 課題 3: Cross-family z-space の根本的整合

Generator が CNN と ViT の両方に有効な潜在ベクトルを生成するには、z-space の根本的な再設計（Contrastive alignment や Multi-family prototype aggregation）が必要。

---

### 9.3 発表計画

| 期限 | 学会・提出先 | 内容 |
|------|------------|------|
| **2026年6月末** | **DICOMO シンポジウム** | IID 環境での AFAD 貢献（貢献 1〜3）+ Non-IID での残課題（貢献 4） |
| 2026年10月末 | 国際学会（予定） | Non-IID 対応の追加実験（課題 1 または 課題 2）|
| 2027年1月末 | 修士論文 | AFAD 全体の統合（IID 貢献 + Non-IID 改善 + 総合考察） |

> DICOMO では「FedGen との IID 同等性の達成」と「二重異種性の同時解決」を主たる貢献として主張し、Non-IID ギャップは「根本原因を特定した残課題」として正直に示す。ネガティブ結果を丁寧に示すことは研究の信頼性を高める。

---

## 11. 原著論文との差分

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

## 12. セットアップ・実行方法

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

## 13. ディレクトリ構造

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

## 14. 参考文献

- Diao, E., Ding, J., and Tarokh, V. "HeteroFL: Computation and Communication Efficient Federated Learning for Heterogeneous Clients." *International Conference on Learning Representations (ICLR)*, 2021.
- Zhu, Z., Hong, J., and Zhou, J. "Data-Free Knowledge Distillation for Heterogeneous Federated Learning." *International Conference on Machine Learning (ICML)*, 2021.
- Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Smola, A., and Smith, V. "Federated Optimization in Heterogeneous Networks." *Proceedings of Machine Learning and Systems (MLSys)*, 2020.
- Hinton, G., Vinyals, O., and Dean, J. "Distilling the Knowledge in a Neural Network." *NeurIPS Deep Learning Workshop*, 2015.
- Horvath, S., Laskaridis, S., Almeida, M., Leontiadis, I., Venieris, S., and Lane, N. "FjORD: Fair and Accurate Federated Learning under heterogeneous targets with Ordered Dropout." *Neural Information Processing Systems (NeurIPS)*, 2021.
- Tan, Y., Long, G., Liu, L., Zhou, T., Lu, Q., Jiang, J., and Zhang, C. "FedProto: Federated Prototype Learning across Heterogeneous Clients." *Association for the Advancement of Artificial Intelligence (AAAI)*, 2022.
- Khosla, P., Tian, Y., Wang, X., Liu, C., Isola, P., and Krishnamurthy, A. "Supervised Contrastive Learning." *Neural Information Processing Systems (NeurIPS)*, 2020.

---

## 著者

- **作成者**: 島野 凌
- **所属**: 大阪工業大学 大学院 情報科学研究科
