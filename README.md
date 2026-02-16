# AFAD (Adaptive Federated Architecture Distribution)

## 概要

連合学習 (Federated Learning; FL) では、参加デバイスの計算資源が不均一であることが一般的であり、
全デバイスに同一モデルを配布する従来の FedAvg では非効率が生じる。
この問題に対して、**異種アーキテクチャ間の効率的な知識共有**を可能にするハイブリッドフレームワーク **AFAD** を提案する。

AFAD は 2 つの既存手法を統合し、それぞれの適用範囲を使い分ける:

- **同族間 (Intra-Family)**: HeteroFL [Diao+, ICLR 2021] による部分重み共有
  （例: ResNet50 ↔ ResNet18 — パラメータ空間が重なる同系列モデル間で直接重みを集約）
- **異種間 (Inter-Family)**: FedGen [Zhu+, ICML 2021] による Data-Free Knowledge Distillation
  （例: CNN ↔ ViT — パラメータ空間が完全に異なるモデル間で合成データを介して知識を転写）

さらに、蒸留プロセスの安定化のために **サーバーサイド蒸留**、**品質ゲート**、**EMA ブレンディング**、**FedProx 正則化** 等の安定化機構を導入し、Non-IID 環境でもロバストな学習を実現する。

---

## 提案手法の詳細

### 1. 問題設定

$K$ 台のクライアントがそれぞれ異なるモデルアーキテクチャ $f_{k}$ と局所データ $\mathcal{D}_{k}$ を持つ連合学習環境を考える。各データは Non-IID に分布しており（Dirichlet 分割、$\alpha = 0.5$）、クライアント間でラベル分布が大きく偏る。

本フレームワークでは以下の 5 種類のモデルを使用する:

| Family | モデル | パラメータ数 | 特徴 |
|--------|--------|:---:|------|
| CNN | ResNet50 | ~23.5M | 深い残差ネットワーク |
| CNN | ResNet18 | ~11.2M | 軽量残差ネットワーク |
| CNN | MobileNetV3-Large | ~4.2M | 逆残差ブロック、モバイル向け |
| ViT | ViT-Tiny | ~5.5M | 自己注意機構によるパッチ処理 |
| ViT | DeiT-Small | ~21.7M | 蒸留トークン付き ViT |

CNN 系と ViT 系はパラメータ空間が根本的に異なる（畳み込みカーネル vs 自己注意重み行列）ため、従来の重み平均では知識共有ができない。これが AFAD のハイブリッドアプローチを必要とする根本的な動機である。

### 2. ラウンドごとの処理フロー

各連合学習ラウンド $t$ は以下の 4 ステップで構成される:

```
Round t:
  [Step 1] Server → Clients : モデルパラメータ配信
  [Step 2] Clients          : ローカル学習 (SGD + FedProx)
  [Step 3] Clients → Server : 更新済みパラメータ送信
  [Step 4] Server           : 集約 (HeteroFL) → Generator 学習 → 知識蒸留 (FedGen)
```

### 3. Step 1–3: ローカル学習と FedProx 正則化

各クライアント $k$ は、受信したグローバルモデルパラメータ $w^{t}$ を初期値として、局所データ $\mathcal{D}_{k}$ 上で SGD による学習を行う。Non-IID 環境でのクライアントドリフト（各クライアントの更新が全体最適から乖離する現象）を抑制するため、FedProx [Li+, MLSys 2020] の近接項を損失関数に追加する:

$$\mathcal{L}_{k} = \mathcal{L}_{CE}(f_{k}(x), y) + \frac{\mu}{2} \| w_{k} - w^{t} \|^2$$

ここで $\mu = 0.01$ は近接項の強度を制御するハイパーパラメータである。第 2 項がグローバルモデルからの過度な乖離を抑制し、Non-IID 環境での学習を安定化させる。

学習率はサーバーが Cosine Annealing でラウンドごとに管理し、全クライアントに配信する:

$$\text{lr}(t) = \text{lr}_{min} + \frac{1}{2}(\text{lr}_{max} - \text{lr}_{min})\left(1 + \cos\left(\frac{\pi \cdot t}{T}\right)\right)$$

ここで $\text{lr}_{max} = 0.01$, $\text{lr}_{min} = 0.0001$, $T$ は総ラウンド数である。

### 4. Step 4a: HeteroFL による同族間集約

クライアントから返送されたモデルパラメータを、**パラメータ形状の一致するグループ（signature group）** ごとに分類し、グループ内でカウントベースの重み平均を行う。

HeteroFL の核心的アイデアは、異なるサイズのモデルであっても **パラメータ空間の先頭部分を共有** できる点にある。例えば ResNet50 の各層の先頭 $r$ 割（$r$ = model_rate）が ResNet18 のパラメータに対応する。集約時には各パラメータ位置について更新されたクライアント数で割る（サンプル数による重み付けではなく均等平均）:

$$w^{t+1}[i,j] = \frac{1}{|S_{i,j}|} \sum_{k \in S_{i,j}} w_{k}^{t}[i,j]$$

ここで $S_{i,j}$ はパラメータ位置 $(i,j)$ を更新したクライアントの集合である。更新されなかった位置はグローバルモデルの値をそのまま保持する。

出力層（分類層）は幅スケーリングの対象外とし、全クライアントが全クラスの分類器を保持する。

### 5. Step 4b: FedGen によるサーバーサイド知識蒸留

HeteroFL は同族間（同じ系列のモデル）でのみ有効であり、CNN と ViT の間では重みの直接共有ができない。この異種間の知識転写を **合成データを介した Data-Free Knowledge Distillation** で実現する。

原論文の FedGen はクライアント側で蒸留を行うが、合成データの品質が低い段階ではクライアントの学習を不安定化させるリスクがある。AFAD ではこの蒸留を **サーバーサイドで実行** し、複数の安定化機構を導入した。

#### 5.1 条件付き Generator の学習

サーバー上の Generator $G$ はクラスラベル $y$ とノイズ $\epsilon$ を入力とし、合成画像 $\hat{x} = G(y, \epsilon)$ を生成する。Generator は全クライアントモデルのアンサンブルが合成画像に対して正しいクラスを一致して予測するよう最適化される:

$$\mathcal{L}_{G} = \alpha \cdot \mathcal{L}_{teacher} + \eta \cdot \mathcal{L}_{diversity}$$

**教師損失** $\mathcal{L}_{teacher}$ は、各モデル $f_m$ がラベル $y$ を正しく予測するよう重み付き交差エントロピーを計算する。重みはクライアントの各ラベルのサンプル数に基づき、Non-IID 環境でのラベル偏りを補正する:

$$\mathcal{L}_{teacher} = \sum_{m=1}^{M} \sum_{i=1}^{B} w_{y_i, m} \cdot \text{CE}(f_m(\hat{x}_i), y_i)$$

**多様性損失** $\mathcal{L}_{diversity}$ は Generator のモード崩壊を防止する。異なるノイズ入力から異なる出力を生成するよう促す:

$$\mathcal{L}_{diversity} = \exp\left(-\text{mean}(d(\hat{x}) \odot d(\epsilon))\right)$$

ここで $d(\cdot)$ はペアワイズ L2 距離である。

#### 5.2 品質ゲート

Generator の学習初期は合成画像の品質が低い。低品質な合成データで蒸留を行うとモデル精度が劣化するため、**品質ゲート** を設けて蒸留の実行可否を判断する:

> アンサンブルの合成画像に対する Top-1 精度が **40% 未満** の場合、そのラウンドの蒸留をスキップする。

この閾値は Generator の成熟度を反映し、十分な品質に達してから蒸留を開始する。

#### 5.3 知識蒸留 (Knowledge Distillation)

品質ゲートを通過した後、各モデルを **生徒 (student)** として、残りの全モデルのアンサンブルを **教師 (teacher)** として知識蒸留を行う。Generator が生成した合成画像 $\hat{x}$ に対して:

1. 教師アンサンブルの soft label を計算: $p_{teacher} = \frac{1}{|T|}\sum_{m \in T} \text{softmax}(f_m(\hat{x}) / \tau)$
2. 生徒の出力との KL ダイバージェンスを最小化: $\mathcal{L}_{KD} = \text{KL}(\log\text{softmax}(f_{student}(\hat{x}) / \tau) \| p_{teacher})$

温度パラメータ $\tau = 4.0$ により、soft label の確率分布を平滑化し、暗黙知（dark knowledge）の転写を促進する。

#### 5.4 EMA ブレンディング

蒸留は合成データ上で行われるため、実データで学習した表現を過度に書き換えるリスクがある。これを防ぐため、蒸留後の重みを元の重みと **指数移動平均 (EMA)** で混合する:

$$w_{new} = (1 - \beta) \cdot w_{original} + \beta \cdot w_{distilled} \quad (\beta = 0.1)$$

$\beta = 0.1$ は蒸留の影響を 10% に抑え、実データで獲得した特徴表現の 90% を保護する。

#### 5.5 Warmup と周期的蒸留

- **Warmup**: 最初の 3 ラウンドは蒸留を行わない。モデルがランダム初期化に近い段階では Generator の学習が不安定なため、十分な精度に達してから蒸留を開始する
- **周期的実行**: 蒸留は warmup 後、2 ラウンドに 1 回の頻度で実行する。毎ラウンド蒸留すると合成データへの過適合による累積的な精度劣化が生じるため、実データ学習と蒸留を交互に行う

#### 5.6 原著論文との設計差分

AFAD の蒸留機構は FedGen [Zhu+, ICML 2021] を出発点としつつ、異種アーキテクチャ対応のために複数の適応を加えている。以下に主要な差分と各設計判断の根拠を示す。

| 観点 | 原著 FedGen | AFAD |
|------|-----------|------|
| Generator 出力 | 潜在ベクトル (32 次元) | 画像 (1×28×28) |
| KD 実行場所 | クライアント側正則化 | サーバーサイド蒸留 |
| KD 損失関数 | CE + KL（温度なし） | T²·KL（T=4.0、Hinton et al., 2015） |
| α, β | 各 10.0、ラウンドごとに 0.98 で減衰 | α=1.0, β=0.1 (EMA)、固定 |
| 集約 | FedAvg（含む） | HeteroFL（別フラグで制御） |
| 品質ゲート | なし | ensemble_acc ≥ 40% |
| EMA ブレンディング | なし | (1-β)·original + β·distilled |

**設計判断の根拠:**

- **画像空間 Generator**: 原著の潜在ベクトル方式は、全クライアントが同一アーキテクチャ（共通の中間層構造）を前提とする。CNN と ViT では中間表現の次元・構造が根本的に異なるため、任意のモデルが消費できる画像空間での生成に変更した
- **サーバーサイド KD**: クライアント側で低品質な合成データを用いた正則化を行うと、実データでの学習を不安定化させるリスクがある。サーバーサイドに移行し、品質ゲートと組み合わせることで、十分な品質の合成データのみを蒸留に使用する
- **T² スケーリング**: Hinton et al. (2015) の標準的な KD 公式に準拠。softmax(·/T) の勾配は 1/T² でスケールされるため、T² を乗じて CE と同等の勾配スケールを回復する
- **EMA ブレンディング**: 合成データ上の蒸留は実データで獲得した特徴表現を上書きするリスクがある。β=0.1 の EMA で蒸留の影響を 10% に制限し、実データ表現の 90% を保護する
- **品質ゲート**: Generator の学習初期は合成画像の品質が低く、この段階で蒸留を行うとモデル精度が劣化する。アンサンブル精度 40% 以上を閾値として蒸留の開始を制御する

> **Note**: 上記の差分により、本実装の「KD Only」モード（`enable_heterofl=False`）は原著 FedGen とは異なるアルゴリズムである。原著 FedGen は FedAvg 集約を含むが、KD Only モードは集約を行わずサーバーサイド蒸留のみを実行する。

### 6. システムアーキテクチャ

```
Server (AFADStrategy)
│
├─ configure_fit()
│   └─ 各クライアントに対応する signature の
│      グローバルモデルと Cosine LR を配信
│
├─ aggregate_fit()
│   ├─ [1] Signature 分類
│   │     クライアントのパラメータ形状から自動グループ化
│   │     例: ResNet50×2 / ResNet18×2 / MNetV3×2 / ViT-Tiny×2 / DeiT-Small×2
│   │
│   ├─ [2] HeteroFL 集約 (グループごと)
│   │     グループ内でカウントベース重み平均
│   │     → 5 つのグローバルモデルを更新
│   │
│   └─ [3] FedGen (全モデル横断)
│         ├─ Generator 学習: 全 5 モデルのアンサンブル一致を最大化
│         ├─ 品質ゲート: ensemble_acc ≥ 40% を確認
│         └─ 知識蒸留: 各モデルを生徒、残り 4 モデルを教師として KD
│            最後に EMA ブレンディングで重みを混合
│
├─ configure_evaluate()
│   └─ 各クライアントに対応するグローバルモデルを配信
│
└─ aggregate_evaluate()
    └─ 加重平均で全体精度を集約

Clients (AFADClient)
├─ Client 0,1: ResNet50   (CNN)     ─┐
├─ Client 2,3: MobileNetV3 (CNN)     ├─ CNN Family
├─ Client 4,5: ResNet18   (CNN)     ─┘
├─ Client 6,7: ViT-Tiny   (ViT)     ─┐
└─ Client 8,9: DeiT-Small (ViT)     ─┘ ViT Family

各クライアント:
  SGD + FedProx (μ=0.01) + Gradient Clipping (max_norm=1.0)
  ローカルエポック数: 3
```

## Phase 1 実験結果

MNIST・IID 分布・5 クライアント・40 ラウンド・Cosine LR での 3 方式比較:

| 方式 | Best Accuracy | Final Accuracy | Total Time |
|------|:---:|:---:|:---:|
| **HeteroFL Only** | **95.58%** | **95.56%** | 1,951s |
| KD Only | 95.37% | 95.31% | 2,161s |
| AFAD Hybrid | 95.52% | 95.48% | 2,210s |

精度推移:

```
         HeteroFL Only    KD Only    AFAD Hybrid
Round  1:    49.41%          50.01%        47.72%
Round  5:    89.69%          89.20%        89.87%
Round 10:    92.96%          93.31%        92.52%
Round 20:    94.73%          94.77%        94.62%
Round 30:    95.43%          95.00%        95.48%
Round 40:    95.56%          95.31%        95.48%
```

> **Note**: 5 つの固有アーキテクチャ構成では、各クライアントが独自の signature グループを形成するため、HeteroFL 集約は単純コピーに退化する。FedGen による知識蒸留が異種モデル間の唯一の知識共有メカニズムとして機能している。3 方式とも 95% 以上の精度に収束し、安定した学習曲線を示す。

## Phase 2 実験結果

OrganAMNIST（11 クラス臓器分類）・Non-IID 分布（Dirichlet α=0.5）・10 クライアント（5 アーキテクチャ × 2）・40 ラウンド・Cosine LR・FedProx (μ=0.01)。

### 3 方式独立比較 (seed=42)

| 方式 | Best Accuracy | Final Accuracy | Loss | Total Time |
|------|:---:|:---:|:---:|:---:|
| HeteroFL Only | 70.91% | 70.78% | 1.4937 | 2,093s |
| KD Only | 57.61% | 57.43% | 2.3869 | 2,450s |
| **AFAD Hybrid** | **71.11%** | **71.11%** | **1.4133** | 2,227s |

### 精度推移

```
         HeteroFL Only    KD Only    AFAD Hybrid
Round  1:    16.07%          35.98%        17.07%
Round  5:    45.53%          45.59%        46.48%
Round 10:    61.55%          52.15%        61.51%
Round 20:    69.07%          55.78%        68.40%
Round 30:    70.85%          57.10%        70.81%
Round 40:    70.78%          57.43%        71.11%
```

> **Phase 2 の主な知見**:
>
> 1. **KD Only は集約なしで大幅に精度低下** (57.61% vs HeteroFL 70.91%, −13.3pt)。
>    KD のみでは同族間の重み共有を代替できず、各クライアントが独立に学習する状態に近い
> 2. **AFAD Hybrid が最高精度** (71.11%) を達成。HeteroFL Only (+0.20pt) を僅かに上回る
> 3. **HeteroFL 集約が支配的要因**: 同族内の重み平均が精度を大きく左右し、FedGen の KD は補助的な役割
> 4. **FedGen の KD 効果は限定的だが正**: AFAD (71.11%) > HeteroFL (70.91%) の差 (+0.20pt) が KD の純粋な貢献分
> 5. **損失値**: KD Only は loss=2.39 と高止まりし、収束が不十分。HeteroFL/AFAD は loss≈1.4 で安定収束
>
> **注**: 本結果は単一 seed での予備的な結果である。Multi-seed 統計的検証を実施予定。

## 考察と限界

### 現時点の評価

修正後の独立 3 方式比較（単一 seed）により、各手法の役割が明確に分離された:

| 側面 | 評価 |
|------|------|
| HeteroFL 集約 | 支配的要因。集約ありの 2 方式 (70.9–71.1%) vs 集約なし (57.6%) で **13pt** の差 |
| FedGen KD | 正だが限定的。AFAD − HeteroFL = **+0.20pt** が KD の純粋な貢献分 |
| AFAD 統合 | 最高精度 (71.11%) を達成するが、HeteroFL 単体との差は僅差 |
| 計算コスト | KD Only は最遅 (2,450s)。集約なしで KD のみは非効率 |
| IID 環境 | Phase 1 では 3 方式ほぼ同等 (95.3–95.6%)。AFAD 固有の優位性は見られない |

### 各手法の独立した効果

Phase 2 の結果から、以下が実験的に確認された:

1. **HeteroFL の効果**: 同族間の重み平均により、個別学習に対して **+13pt** の精度向上。これは同じ signature を持つクライアント間で学習した表現を直接共有できるため
2. **FedGen の効果**: HeteroFL 集約に KD を追加することで **+0.20pt** の改善。CNN ↔ ViT 間の知識転写が少なくとも精度を劣化させず、わずかに寄与
3. **FedGen 単体の限界**: 集約なし（各クライアント独立）+ KD のみでは 57.6% に留まり、KD だけでは同族間の重み共有を代替できないことが明確

### 限界と今後の課題

1. **単一 seed での予備結果**: Multi-seed 統計的検証が必要。AFAD vs HeteroFL の +0.20pt が有意かどうかは未確定
2. **Non-IID 強度の拡張**: α=0.1 等のより強い Non-IID 環境で、KD の寄与が拡大するか検証
3. **スケール検証**: クライアント数を 20–50 に増加し、同族グループ内の集約効果を強化した場合の評価
4. **データセットの多様化**: CIFAR-10/100、PathMNIST 等の異なるタスクでの再現性検証
5. **長期的収束**: ラウンド数を 100 以上に拡張し、KD の累積効果が顕在化するか評価
6. **多角的評価**: 精度に加え、通信コスト、収束速度、クライアント間の公平性を評価指標に追加

### AFAD の意義

AFAD の主たる貢献は精度の絶対的向上ではなく、**異種アーキテクチャが混在する現実的な FL 環境において、同族間集約（HeteroFL）と異種間知識転写（FedGen）を統一的に扱える汎用フレームワークを提示した点**にある。

実験結果は以下の構造的知見を提供する:

- **HeteroFL 集約は不可欠**: 同族間の重み平均がない KD Only は 57.6% に留まり、KD だけでは集約を代替できない
- **FedGen KD は補助的だが正の効果**: AFAD が HeteroFL を +0.20pt 上回り、異種間知識転写が少なくとも精度を劣化させない
- **統合の妥当性**: 両機構が独立した役割を持ち、AFAD はモデル構成の制約なく連合学習を運用できる汎用フレームワークとして機能する

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

# Multi-seed 統計的検証 (10 seeds, 全30実験, 約17時間)
python scripts/run_multi_seed.py config/afad_phase2_config.yaml

# seed を指定して実行
python scripts/run_multi_seed.py config/afad_phase2_config.yaml --seeds 42 123 456

# 出力先を指定
python scripts/run_multi_seed.py config/afad_phase2_config.yaml --output results/my_results.json
```

> **Note**: `run_multi_seed.py` は `uv run` ではなく `python` (venv 直接) で実行すること。Ray ワーカーの runtime_env 構築問題を回避するため。結果は JSON に逐次保存され、中断後の再実行で完了済み seed を自動スキップする。

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

## ディレクトリ構造

```
AFAD/
├── config/
│   ├── afad_config.yaml          # Phase 1 実験設定 (MNIST)
│   └── afad_phase2_config.yaml   # Phase 2 実験設定 (OrganAMNIST)
├── scripts/
│   ├── run_experiment.py         # 単一実験スクリプト
│   ├── run_comparison.py         # 3 方式比較スクリプト
│   ├── run_multi_seed.py         # Multi-seed 統計的検証スクリプト
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
├── tests/                        # テスト (84 件)
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

## 参考文献

- Diao, E. et al. "HeteroFL: Computation and Communication Efficient Federated Learning for Heterogeneous Clients" (ICLR 2021)
- Zhu, Z. et al. "Data-Free Knowledge Distillation for Heterogeneous Federated Learning" (ICML 2021)
- Li, T. et al. "Federated Optimization in Heterogeneous Networks" (MLSys 2020)

## 開発者

- **作成者**: 島野 凌
- **所属**: 大阪工業大学 大学院 情報科学研究科
