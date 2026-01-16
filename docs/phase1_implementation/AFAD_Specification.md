# AFAD (Adaptive Federated Architecture Distribution) 機能・外部仕様書

**バージョン**: 1.0  
**作成日**: 2025年1月16日  
**作成者**: 島野 凌  
**所属**: 大阪工業大学 大学院 情報科学研究科

---

## 目次

1. [概要](#1-概要)
2. [システム要件](#2-システム要件)
3. [システム構成](#3-システム構成)
4. [機能要件](#4-機能要件)
5. [クラス設計](#5-クラス設計)
6. [シーケンス設計](#6-シーケンス設計)
7. [データ構造](#7-データ構造)
8. [API仕様](#8-api仕様)
9. [評価計画](#9-評価計画)
10. [実装フェーズ](#10-実装フェーズ)

---

## 1. 概要

### 1.1 目的

AFAD（Adaptive Federated Architecture Distribution）は、連合学習におけるモデル異質性問題を解決するハイブリッドフレームワークである。本仕様書は、FedMD（知識蒸留系）とHeteroFL（部分学習系）を組み合わせた実装の設計ドキュメントとして機能する。

### 1.2 解決する課題

| 課題 | 従来手法の問題 | AFADの解決策 |
|------|---------------|-------------|
| モデル異質性 | FedAvgは同一モデル構造が必須 | 異種アーキテクチャ間の知識共有を実現 |
| 公開データ依存 | FedMDは公開データセットが必須 | Generator-based Data-Free KDで解消 |
| 計算資源格差 | 低スペック端末の排除 | 動的モデル割当で全端末を包摂 |

### 1.3 コアアプローチ

1. **同族間（Family-Internal）**: HeteroFL方式の部分的重み共有
2. **異種間（Cross-Family）**: Data-Free知識蒸留（FedGen方式）
3. **動的モデル割当**: サーバー主導でクライアント資源に応じたモデル配布

### 1.4 スコープ

**Phase 1（動作実証）**:
- データセット: MNIST → MedMNIST
- クライアント数: 5台（仮想）
- データ分布: IID

**Phase 2（精度向上）**:
- Non-IID対応
- 医療画像での評価
- 通信効率最適化

---

## 2. システム要件

### 2.1 ハードウェア要件

| マシン | OS | GPU | 役割 |
|--------|-----|-----|------|
| Server Machine | Ubuntu 22.04+ | RTX 4090 | FLサーバー + 仮想クライアント×2 |
| Client Machine | Windows (WSL2) | RTX 4090 | 仮想クライアント×3 |

### 2.2 ソフトウェア要件

```yaml
# 必須パッケージ
python: ">=3.10"
pytorch: ">=2.0.0"
torchvision: ">=0.15.0"
flwr: ">=1.7.0"  # Flower Framework
numpy: ">=1.24.0"
scipy: ">=1.10.0"

# 補助パッケージ
medmnist: ">=2.2.0"
wandb: ">=0.15.0"       # 実験ログ（オプション）
matplotlib: ">=3.7.0"
pyyaml: ">=6.0"
```

### 2.3 ネットワーク要件

- サーバー・クライアント間: TCP/IP通信
- 推奨帯域: 100Mbps以上
- ポート: 8080（Flowerデフォルト）

---

## 3. システム構成

### 3.1 システム構成図

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AFAD System Architecture                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         Server (Ubuntu + RTX 4090)                   │    │
│  │  ┌─────────────────────────────────────────────────────────────┐   │    │
│  │  │                    AFADServer                                │   │    │
│  │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │   │    │
│  │  │  │  Generator   │  │  Family      │  │  Aggregation     │  │   │    │
│  │  │  │  (Data-Free  │  │  Router      │  │  Manager         │  │   │    │
│  │  │  │   KD用)      │  │              │  │                  │  │   │    │
│  │  │  └──────────────┘  └──────────────┘  └──────────────────┘  │   │    │
│  │  │                                                              │   │    │
│  │  │  ┌──────────────────────────────────────────────────────┐   │   │    │
│  │  │  │              AFADStrategy (Flower Strategy)          │   │   │    │
│  │  │  │  - configure_fit()    - aggregate_fit()              │   │   │    │
│  │  │  │  - configure_evaluate() - aggregate_evaluate()       │   │   │    │
│  │  │  └──────────────────────────────────────────────────────┘   │   │    │
│  │  └─────────────────────────────────────────────────────────────┘   │    │
│  │                                                                     │    │
│  │  ┌─────────────────┐  ┌─────────────────┐                         │    │
│  │  │ Virtual Client 0│  │ Virtual Client 1│  (CNN Family)           │    │
│  │  │ ResNet50        │  │ MobileNetV3     │                         │    │
│  │  └─────────────────┘  └─────────────────┘                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                      │                                       │
│                                      │ gRPC (TCP:8080)                       │
│                                      │                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      Client Machine (WSL2 + RTX 4090)               │    │
│  │                                                                      │    │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │    │
│  │  │ Virtual Client 2│  │ Virtual Client 3│  │ Virtual Client 4│    │    │
│  │  │ ResNet18        │  │ ViT-Tiny        │  │ DeiT-Small      │    │    │
│  │  │ (CNN Family)    │  │ (ViT Family)    │  │ (ViT Family)    │    │    │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 モデルファミリー構成

#### CNN Family（同族：重み共有可能）

| モデル | パラメータ数 | 想定デバイス | 複雑度レベル |
|--------|-------------|-------------|-------------|
| ResNet50 | 25.6M | データセンター | a (100%) |
| ResNet34 | 21.8M | ワークステーション | b (85%) |
| ResNet18 | 11.7M | デスクトップPC | c (46%) |
| MobileNetV3-Large | 5.4M | ノートPC | d (21%) |
| MobileNetV2 | 3.4M | エッジデバイス | e (13%) |

#### ViT Family（同族：重み共有可能）

| モデル | パラメータ数 | 想定デバイス | 複雑度レベル |
|--------|-------------|-------------|-------------|
| ViT-Base | 86M | データセンター | a (100%) |
| DeiT-Small | 22M | ワークステーション | b (26%) |
| ViT-Small | 22M | デスクトップPC | c (26%) |
| DeiT-Tiny | 5M | ノートPC | d (6%) |
| ViT-Tiny | 5M | エッジデバイス | e (6%) |

---

## 4. 機能要件

### 4.1 機能一覧

| ID | 機能名 | 説明 | 優先度 |
|----|--------|------|--------|
| F-001 | アーキテクチャファミリー検出 | クライアントモデルの構造を解析し、CNN/ViT等のファミリーに分類 | 高 |
| F-002 | 同族間重み集約 | HeteroFL方式による部分的パラメータ共有と集約 | 高 |
| F-003 | 合成データ生成 | FedGen方式のGeneratorによるData-Free KD用データ生成 | 高 |
| F-004 | 異種間知識蒸留 | CNN↔ViT間のロジットベース知識転移 | 高 |
| F-005 | 動的モデル割当 | クライアント資源に応じた最適モデルの選択・配布 | 中 |
| F-006 | 特徴量アライメント | CKA-basedのクロスアーキテクチャ特徴整合 | 中 |
| F-007 | 評価メトリクス計算 | Accuracy, F1-score, AUC-ROC, 通信コストの計測 | 高 |

### 4.2 機能詳細

#### F-001: アーキテクチャファミリー検出

**入力**: PyTorchモデル (`nn.Module`)  
**出力**: ファミリーID (`str`), 複雑度レベル (`float`)

**検出ロジック**:
```python
検出パターン = {
    "cnn": ["Conv2d", "BatchNorm2d", "ReLU", "MaxPool2d"],
    "resnet": ["Conv2d", "BatchNorm2d", "残差接続"],
    "vit": ["PatchEmbed", "MultiheadAttention", "LayerNorm", "MLP"]
}
```

#### F-002: 同族間重み集約（HeteroFL方式）

**処理フロー**:
1. 各クライアントから部分パラメータを受信
2. チャンネル数に基づく選択的平均化
3. グローバルモデルの該当部分を更新

**数式**:
$$W_{global}[0:\beta K] = \frac{1}{|S|} \sum_{i \in S} W_i[0:\beta_i K]$$

ここで $\beta_i$ はクライアント $i$ の複雑度レベル、$K$ は最大チャンネル数

#### F-003: 合成データ生成（FedGen方式）

**Generator構造**:
- 入力: ノイズ $z \sim \mathcal{N}(0, I)$ + ラベル $y$
- 出力: 特徴表現 $f \in \mathbb{R}^{d}$（画像ではなく潜在表現）

**学習目標**:
$$\min_G \mathbb{E}_{z,y} \left[ \mathcal{L}_{CE}\left( \sigma\left(\frac{1}{K}\sum_k g_k(G(z,y))\right), y \right) \right]$$

#### F-004: 異種間知識蒸留

**損失関数**:
$$\mathcal{L}_{KD} = \tau^2 \cdot KL\left( \sigma\left(\frac{z_s}{\tau}\right) \| \sigma\left(\frac{z_t}{\tau}\right) \right)$$

- $z_s$: Student（個別モデル）の出力ロジット
- $z_t$: Teacher（アンサンブル）の出力ロジット
- $\tau$: 温度パラメータ（推奨: 4.0〜6.0）

---

## 5. クラス設計

### 5.1 クラス図

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Class Diagram                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────┐      │
│  │                    <<interface>>                                   │      │
│  │                    AggregationStrategy                             │      │
│  ├───────────────────────────────────────────────────────────────────┤      │
│  │ + aggregate(updates: List[ModelUpdate]) -> GlobalModel            │      │
│  │ + configure(config: Dict) -> None                                 │      │
│  └───────────────────────────────────────────────────────────────────┘      │
│                              △                                               │
│                              │                                               │
│          ┌──────────────────┼──────────────────┐                            │
│          │                  │                  │                            │
│  ┌───────┴───────┐  ┌───────┴───────┐  ┌──────┴────────┐                   │
│  │ HeteroFL      │  │ FedMDStrategy │  │ AFADStrategy  │                   │
│  │ Strategy      │  │               │  │               │                   │
│  ├───────────────┤  ├───────────────┤  ├───────────────┤                   │
│  │ - capacities  │  │ - temperature │  │ - generator   │                   │
│  │ - width_ratio │  │ - public_data │  │ - family_router│                  │
│  ├───────────────┤  ├───────────────┤  ├───────────────┤                   │
│  │ + selective_  │  │ + distill()   │  │ + hybrid_     │                   │
│  │   average()   │  │ + compute_    │  │   aggregate() │                   │
│  │               │  │   ensemble()  │  │ + cross_family│                   │
│  │               │  │               │  │   _distill()  │                   │
│  └───────────────┘  └───────────────┘  └───────┬───────┘                   │
│                                                 │                            │
│                                                 │ uses                       │
│                                                 ▼                            │
│  ┌───────────────────────────────────────────────────────────────────┐      │
│  │                        FamilyRouter                                │      │
│  ├───────────────────────────────────────────────────────────────────┤      │
│  │ - family_signatures: Dict[str, List[str]]                         │      │
│  │ - registered_clients: Dict[str, FamilyInfo]                       │      │
│  ├───────────────────────────────────────────────────────────────────┤      │
│  │ + detect_family(model: nn.Module) -> str                          │      │
│  │ + get_complexity_level(model: nn.Module) -> float                 │      │
│  │ + is_same_family(model1, model2) -> bool                          │      │
│  │ + register_client(client_id: str, model: nn.Module) -> None       │      │
│  │ + get_family_members(family: str) -> List[str]                    │      │
│  └───────────────────────────────────────────────────────────────────┘      │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────┐      │
│  │                     SyntheticGenerator                             │      │
│  ├───────────────────────────────────────────────────────────────────┤      │
│  │ - latent_dim: int                                                 │      │
│  │ - num_classes: int                                                │      │
│  │ - hidden_dims: List[int]                                          │      │
│  │ - ema_generator: nn.Module  (DFRD方式)                            │      │
│  │ - ema_decay: float                                                │      │
│  ├───────────────────────────────────────────────────────────────────┤      │
│  │ + forward(z: Tensor, labels: Tensor) -> Tensor                    │      │
│  │ + generate_batch(batch_size: int, labels: Tensor) -> Tensor       │      │
│  │ + update_ema() -> None                                            │      │
│  │ + train_step(ensemble_logits: Tensor) -> float                    │      │
│  └───────────────────────────────────────────────────────────────────┘      │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────┐      │
│  │                       AFADClient                                   │      │
│  ├───────────────────────────────────────────────────────────────────┤      │
│  │ - client_id: str                                                  │      │
│  │ - model: nn.Module                                                │      │
│  │ - family: str                                                     │      │
│  │ - complexity_level: float                                         │      │
│  │ - local_dataset: DataLoader                                       │      │
│  │ - optimizer: Optimizer                                            │      │
│  ├───────────────────────────────────────────────────────────────────┤      │
│  │ + fit(parameters, config) -> Tuple[Parameters, int, Dict]         │      │
│  │ + evaluate(parameters, config) -> Tuple[float, int, Dict]         │      │
│  │ + compute_logits(synthetic_data: Tensor) -> Tensor                │      │
│  │ + local_train(epochs: int) -> float                               │      │
│  │ + distill_from_ensemble(ensemble_logits: Tensor, temp: float)     │      │
│  └───────────────────────────────────────────────────────────────────┘      │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────┐      │
│  │                      ModelRegistry                                 │      │
│  ├───────────────────────────────────────────────────────────────────┤      │
│  │ - models: Dict[str, Callable]                                     │      │
│  │ - family_mapping: Dict[str, str]                                  │      │
│  ├───────────────────────────────────────────────────────────────────┤      │
│  │ + register(name: str, factory: Callable, family: str) -> None     │      │
│  │ + create_model(name: str, num_classes: int) -> nn.Module          │      │
│  │ + get_available_models(family: str) -> List[str]                  │      │
│  │ + get_model_info(name: str) -> ModelInfo                          │      │
│  └───────────────────────────────────────────────────────────────────┘      │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────┐      │
│  │                      MetricsCollector                              │      │
│  ├───────────────────────────────────────────────────────────────────┤      │
│  │ - history: Dict[str, List[float]]                                 │      │
│  │ - communication_costs: List[int]                                  │      │
│  ├───────────────────────────────────────────────────────────────────┤      │
│  │ + record_round(round: int, metrics: Dict) -> None                 │      │
│  │ + compute_accuracy(preds, labels) -> float                        │      │
│  │ + compute_f1_score(preds, labels) -> float                        │      │
│  │ + compute_auc_roc(probs, labels) -> float                         │      │
│  │ + compute_communication_cost(updates: List) -> int                │      │
│  │ + export_results(path: str) -> None                               │      │
│  └───────────────────────────────────────────────────────────────────┘      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 クラス詳細

#### 5.2.1 AFADStrategy

Flower Strategyを継承したAFADのメイン集約クラス。

```python
class AFADStrategy(fl.server.strategy.Strategy):
    """
    AFADのハイブリッド集約戦略
    
    Attributes:
        generator (SyntheticGenerator): Data-Free KD用の合成データ生成器
        family_router (FamilyRouter): アーキテクチャファミリー分類器
        temperature (float): 知識蒸留の温度パラメータ
        distill_steps (int): 蒸留ステップ数/ラウンド
        intra_family_strategy (str): 同族内集約方式 ("heterofl" | "fedavg")
        inter_family_strategy (str): 異種間集約方式 ("fedgen" | "dfrd")
    """
```

**主要メソッド**:

| メソッド | 引数 | 戻り値 | 説明 |
|----------|------|--------|------|
| `configure_fit` | server_round, parameters, client_manager | List[Tuple[ClientProxy, FitIns]] | 学習設定の配布 |
| `aggregate_fit` | server_round, results, failures | Tuple[Parameters, Dict] | ハイブリッド集約の実行 |
| `_intra_family_aggregate` | family, updates | Parameters | 同族内HeteroFL集約 |
| `_inter_family_distill` | family_models | Dict[str, Parameters] | 異種間知識蒸留 |

#### 5.2.2 FamilyRouter

```python
class FamilyRouter:
    """
    モデルアーキテクチャのファミリー分類と管理
    
    Attributes:
        family_signatures (Dict[str, List[str]]): 各ファミリーの構造シグネチャ
        registered_clients (Dict[str, ClientInfo]): 登録済みクライアント情報
    """
    
    SIGNATURES = {
        "resnet": ["Conv2d", "BatchNorm2d", "relu", "residual"],
        "mobilenet": ["Conv2d", "BatchNorm2d", "relu6", "inverted_residual"],
        "vit": ["PatchEmbed", "Attention", "LayerNorm", "Mlp"],
        "deit": ["PatchEmbed", "Attention", "LayerNorm", "dist_token"]
    }
    
    FAMILY_GROUPS = {
        "cnn": ["resnet", "mobilenet", "vgg", "efficientnet"],
        "transformer": ["vit", "deit", "swin"]
    }
```

#### 5.2.3 SyntheticGenerator

```python
class SyntheticGenerator(nn.Module):
    """
    FedGen/DFRD方式の合成データ生成器
    
    Args:
        latent_dim (int): 潜在空間の次元数 (default: 100)
        num_classes (int): クラス数
        feature_dim (int): 出力特徴量の次元数
        ema_decay (float): EMAの減衰率 (default: 0.9)
    """
    
    def __init__(self, latent_dim=100, num_classes=10, feature_dim=512, ema_decay=0.9):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, latent_dim)
        self.generator = nn.Sequential(
            nn.Linear(latent_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )
        self.ema_generator = copy.deepcopy(self.generator)
        self.ema_decay = ema_decay
```

#### 5.2.4 AFADClient

```python
class AFADClient(fl.client.NumPyClient):
    """
    AFADクライアントの実装
    
    Attributes:
        client_id (str): クライアント識別子
        model (nn.Module): ローカルモデル（CNN or ViT）
        family (str): 所属ファミリー
        complexity_level (float): 複雑度レベル (0.0-1.0)
        local_dataset (DataLoader): ローカルデータセット
        device (torch.device): 計算デバイス
    """
    
    def fit(self, parameters, config):
        """
        ローカル学習の実行
        
        Returns:
            parameters: 更新後のモデルパラメータ
            num_examples: 学習サンプル数
            metrics: {"loss": float, "logits": np.ndarray (optional)}
        """
        
    def compute_logits(self, synthetic_data: torch.Tensor) -> np.ndarray:
        """
        合成データに対するロジット計算（異種間KD用）
        """
```

---

## 6. シーケンス設計

### 6.1 1ラウンドの処理フロー

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Sequence Diagram: 1 Round                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Server          FamilyRouter      Generator      CNN_Clients   ViT_Clients │
│    │                  │               │               │              │      │
│    │  1. Start Round  │               │               │              │      │
│    │─────────────────────────────────────────────────────────────────►      │
│    │                  │               │               │              │      │
│    │  2. Generate Synthetic Data      │               │              │      │
│    │─────────────────────────────────►│               │              │      │
│    │                  │               │               │              │      │
│    │  3. Return: z_synthetic          │               │              │      │
│    │◄─────────────────────────────────│               │              │      │
│    │                  │               │               │              │      │
│    │  4. configure_fit (params + z_synthetic)        │              │      │
│    │────────────────────────────────────────────────►│              │      │
│    │────────────────────────────────────────────────────────────────►      │
│    │                  │               │               │              │      │
│    │                  │               │  5. Local Training           │      │
│    │                  │               │  ┌───────────┐│              │      │
│    │                  │               │  │ train()   ││              │      │
│    │                  │               │  └───────────┘│              │      │
│    │                  │               │               │  ┌───────────┐      │
│    │                  │               │               │  │ train()   │      │
│    │                  │               │               │  └───────────┘      │
│    │                  │               │               │              │      │
│    │  6. Return: (params, logits)     │               │              │      │
│    │◄────────────────────────────────────────────────│              │      │
│    │◄──────────────────────────────────────────────────────────────│      │
│    │                  │               │               │              │      │
│    │  7. Classify by Family           │               │              │      │
│    │─────────────────►│               │               │              │      │
│    │                  │               │               │              │      │
│    │  8. Return: family_groups        │               │              │      │
│    │◄─────────────────│               │               │              │      │
│    │                  │               │               │              │      │
│    │  ═══════════════════════════════════════════════════════════   │      │
│    │  ║ Phase 1: Intra-Family Aggregation (HeteroFL)            ║   │      │
│    │  ═══════════════════════════════════════════════════════════   │      │
│    │                  │               │               │              │      │
│    │  9. aggregate_cnn = HeteroFL_avg(cnn_updates)   │              │      │
│    │  10. aggregate_vit = HeteroFL_avg(vit_updates)  │              │      │
│    │                  │               │               │              │      │
│    │  ═══════════════════════════════════════════════════════════   │      │
│    │  ║ Phase 2: Inter-Family Distillation (FedGen)             ║   │      │
│    │  ═══════════════════════════════════════════════════════════   │      │
│    │                  │               │               │              │      │
│    │  11. ensemble_logits = weighted_avg(all_logits) │              │      │
│    │                  │               │               │              │      │
│    │  12. Train Generator             │               │              │      │
│    │─────────────────────────────────►│               │              │      │
│    │  for step in distill_steps:      │               │              │      │
│    │    z = Generator(noise, label)   │               │              │      │
│    │    loss = CE(ensemble(z), label) │               │              │      │
│    │    Generator.update()            │               │              │      │
│    │                  │               │               │              │      │
│    │  13. Distill to Family Models    │               │              │      │
│    │  for family_model in [cnn, vit]: │               │              │      │
│    │    z = Generator(noise)          │               │              │      │
│    │    loss = KL(family_model(z), ensemble_logits)  │              │      │
│    │    family_model.update()         │               │              │      │
│    │                  │               │               │              │      │
│    │  14. End Round                   │               │              │      │
│    │◄────────────────────────────────────────────────────────────   │      │
│    │                  │               │               │              │      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 初期化シーケンス

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Sequence Diagram: Initialization                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Main           Server         ClientManager      Client_0 ... Client_4     │
│   │               │                  │                    │                 │
│   │  1. Load Config               │                    │                 │
│   │──────────────►│                  │                    │                 │
│   │               │                  │                    │                 │
│   │  2. Initialize AFADStrategy      │                    │                 │
│   │──────────────►│                  │                    │                 │
│   │               │                  │                    │                 │
│   │               │  3. Create Generator                  │                 │
│   │               │  4. Create FamilyRouter               │                 │
│   │               │                  │                    │                 │
│   │  5. Start Server (port 8080)     │                    │                 │
│   │──────────────►│                  │                    │                 │
│   │               │                  │                    │                 │
│   │               │                  │  6. Register Clients                 │
│   │               │                  │◄───────────────────│                 │
│   │               │                  │                    │                 │
│   │               │  7. Get Client Info                   │                 │
│   │               │─────────────────►│                    │                 │
│   │               │                  │                    │                 │
│   │               │  8. Detect Family for each client     │                 │
│   │               │  client_0: ResNet50  → CNN Family     │                 │
│   │               │  client_1: MobileNetV3 → CNN Family   │                 │
│   │               │  client_2: ResNet18  → CNN Family     │                 │
│   │               │  client_3: ViT-Tiny  → ViT Family     │                 │
│   │               │  client_4: DeiT-Small → ViT Family    │                 │
│   │               │                  │                    │                 │
│   │               │  9. Initialize Family Models          │                 │
│   │               │  - cnn_global = ResNet50 (template)   │                 │
│   │               │  - vit_global = ViT-Base (template)   │                 │
│   │               │                  │                    │                 │
│   │  10. Ready to Start FL           │                    │                 │
│   │◄──────────────│                  │                    │                 │
│   │               │                  │                    │                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. データ構造

### 7.1 設定ファイル構造

```yaml
# config/afad_config.yaml

experiment:
  name: "AFAD_MNIST_Phase1"
  seed: 42
  num_rounds: 50
  
server:
  address: "0.0.0.0:8080"
  min_clients: 5
  min_fit_clients: 5
  
strategy:
  type: "AFAD"
  intra_family: "heterofl"      # 同族内集約方式
  inter_family: "fedgen"        # 異種間集約方式
  temperature: 4.0              # KD温度
  distill_steps: 10             # 蒸留ステップ数/ラウンド
  generator:
    latent_dim: 100
    hidden_dims: [256, 512]
    ema_decay: 0.9

clients:
  - id: "client_0"
    model: "resnet50"
    device: "cuda:0"
    host: "server"              # Ubuntu machine
  - id: "client_1"
    model: "mobilenetv3_large"
    device: "cuda:0"
    host: "server"
  - id: "client_2"
    model: "resnet18"
    device: "cuda:0"
    host: "wsl2"                # WSL2 machine
  - id: "client_3"
    model: "vit_tiny"
    device: "cuda:0"
    host: "wsl2"
  - id: "client_4"
    model: "deit_small"
    device: "cuda:0"
    host: "wsl2"

data:
  dataset: "mnist"              # Phase1: mnist, Phase2: medmnist
  num_classes: 10
  distribution: "iid"           # iid | non_iid
  batch_size: 64
  
training:
  local_epochs: 5
  learning_rate: 0.01
  optimizer: "sgd"
  momentum: 0.9
  weight_decay: 0.0001

evaluation:
  metrics: ["accuracy", "f1_score", "communication_cost"]
  eval_every: 1                 # ラウンド毎に評価
```

### 7.2 通信データ構造

#### FitIns（サーバー→クライアント）

```python
@dataclass
class AFADFitIns:
    """学習指示データ"""
    parameters: Parameters          # モデルパラメータ（HeteroFL用）
    config: Dict[str, Any] = {
        "local_epochs": int,
        "learning_rate": float,
        "synthetic_data": bytes,    # Pickled tensor (KD用)
        "temperature": float,
        "require_logits": bool,     # ロジット返送要否
    }
```

#### FitRes（クライアント→サーバー）

```python
@dataclass
class AFADFitRes:
    """学習結果データ"""
    parameters: Parameters          # 更新後パラメータ
    num_examples: int               # 学習サンプル数
    metrics: Dict[str, Any] = {
        "loss": float,
        "accuracy": float,
        "logits": bytes,            # Pickled tensor (KD用)
        "family": str,
        "complexity_level": float,
    }
```

### 7.3 評価結果構造

```python
@dataclass
class RoundMetrics:
    """1ラウンドの評価結果"""
    round_number: int
    
    # 全体指標
    global_accuracy: float
    global_f1_score: float
    global_loss: float
    
    # ファミリー別指標
    cnn_family_accuracy: float
    vit_family_accuracy: float
    
    # クライアント別指標
    client_metrics: Dict[str, Dict[str, float]]
    
    # 通信コスト
    upload_bytes: int
    download_bytes: int
    total_communication: int
    
    # 時間
    round_duration_sec: float
```

---

## 8. API仕様

### 8.1 サーバーAPI

#### `AFADServer.start()`

```python
def start(
    config_path: str = "config/afad_config.yaml",
    num_rounds: int = 50
) -> None:
    """
    AFADサーバーを起動
    
    Args:
        config_path: 設定ファイルパス
        num_rounds: 実行ラウンド数
        
    Raises:
        FileNotFoundError: 設定ファイルが見つからない
        ConnectionError: クライアント接続エラー
    """
```

### 8.2 クライアントAPI

#### `AFADClient.fit()`

```python
def fit(
    self,
    parameters: Parameters,
    config: Dict[str, Any]
) -> Tuple[Parameters, int, Dict[str, Any]]:
    """
    ローカル学習を実行
    
    Args:
        parameters: サーバーからのモデルパラメータ
        config: 学習設定
            - local_epochs (int): ローカルエポック数
            - learning_rate (float): 学習率
            - synthetic_data (bytes): KD用合成データ
            - temperature (float): KD温度
            
    Returns:
        Tuple of:
            - Parameters: 更新後パラメータ
            - int: 学習サンプル数
            - Dict: メトリクス（loss, accuracy, logits等）
    """
```

#### `AFADClient.compute_logits()`

```python
def compute_logits(
    self,
    synthetic_data: torch.Tensor,
    temperature: float = 4.0
) -> np.ndarray:
    """
    合成データに対するsoft labelを計算
    
    Args:
        synthetic_data: Generator出力の特徴量 [B, D]
        temperature: Softmax温度
        
    Returns:
        np.ndarray: Soft labels [B, num_classes]
    """
```

### 8.3 Generator API

#### `SyntheticGenerator.generate_batch()`

```python
def generate_batch(
    self,
    batch_size: int,
    labels: Optional[torch.Tensor] = None,
    use_ema: bool = False
) -> torch.Tensor:
    """
    合成データのバッチを生成
    
    Args:
        batch_size: 生成サンプル数
        labels: 条件付けラベル（Noneの場合ランダム）
        use_ema: EMA generatorを使用するか
        
    Returns:
        torch.Tensor: 合成特徴量 [B, feature_dim]
    """
```

#### `SyntheticGenerator.train_step()`

```python
def train_step(
    self,
    ensemble_logits: torch.Tensor,
    labels: torch.Tensor
) -> float:
    """
    Generatorの1ステップ学習
    
    Args:
        ensemble_logits: アンサンブルモデルのロジット [B, C]
        labels: 正解ラベル [B]
        
    Returns:
        float: 学習損失
    """
```

### 8.4 FamilyRouter API

#### `FamilyRouter.detect_family()`

```python
def detect_family(
    self,
    model: nn.Module
) -> Tuple[str, str]:
    """
    モデルのファミリーを検出
    
    Args:
        model: PyTorchモデル
        
    Returns:
        Tuple of:
            - str: 詳細ファミリー ("resnet", "vit", etc.)
            - str: 上位グループ ("cnn", "transformer")
            
    Example:
        >>> router = FamilyRouter()
        >>> family, group = router.detect_family(resnet18())
        >>> print(family, group)
        "resnet" "cnn"
    """
```

#### `FamilyRouter.get_complexity_level()`

```python
def get_complexity_level(
    self,
    model: nn.Module
) -> float:
    """
    モデルの複雑度レベルを計算
    
    Args:
        model: PyTorchモデル
        
    Returns:
        float: 複雑度 (0.0 ~ 1.0)
               1.0 = ファミリー内最大モデル
               
    Note:
        パラメータ数ベースで計算:
        level = params(model) / params(family_max_model)
    """
```

---

## 9. 評価計画

### 9.1 評価指標

| 指標 | 計算式 | 用途 |
|------|--------|------|
| Accuracy | $\frac{TP + TN}{Total}$ | 基本性能 |
| Macro F1-Score | $\frac{1}{C}\sum_{c=1}^{C} F1_c$ | クラス不均衡対応 |
| AUC-ROC | ROC曲線下面積 | 確率出力の品質（医療用） |
| Communication Cost | $\sum_{round} (Upload + Download)$ bytes | 効率性 |
| Convergence Speed | 目標精度到達ラウンド数 | 収束性能 |

### 9.2 ベースライン比較実験

| 実験ID | 手法 | 設定 | 目的 |
|--------|------|------|------|
| EXP-01 | FedAvg | 全クライアント同一モデル（ResNet18） | ベースライン |
| EXP-02 | HeteroFL | CNN Family のみ（ResNet系） | 同族間性能 |
| EXP-03 | FedMD | 公開データあり（MNIST subset） | 異種間性能（理想） |
| EXP-04 | AFAD | フルシステム | 提案手法 |
| EXP-05 | AFAD-ablation | Generator無し（ロジット平均のみ） | アブレーション |

### 9.3 Phase 1 成功基準

| 基準 | 条件 | 判定 |
|------|------|------|
| 動作実証 | 5クライアント50ラウンド完走 | 必須 |
| 基本性能 | MNIST Accuracy ≥ 95% | 必須 |
| ベースライン超過 | FedAvg比 +2% 以上 | 目標 |
| 異種間効果 | CNN単独 vs AFAD で改善 | 目標 |

---

## 10. 実装フェーズ

### 10.1 Phase 1: 動作実証（〜1月末）

```
Week 1-2: 基盤構築
├── [ ] プロジェクト構造の作成
├── [ ] 設定ファイルパーサーの実装
├── [ ] ModelRegistry の実装
├── [ ] 基本的な Flower Server/Client の動作確認
└── [ ] MNIST データローダーの実装

Week 3-4: コアアルゴリズム
├── [ ] FamilyRouter の実装
├── [ ] HeteroFL集約 (同族間) の実装
├── [ ] SyntheticGenerator の実装
├── [ ] 異種間KD の実装
└── [ ] AFADStrategy の統合

Week 5-6: 検証・デバッグ
├── [ ] 単体テストの作成
├── [ ] 5クライアント環境での動作確認
├── [ ] ベースライン実験の実施
├── [ ] バグ修正・最適化
└── [ ] Phase 1 レポート作成
```

### 10.2 Phase 2: 精度向上（2月〜）

```
├── [ ] MedMNIST への移行
├── [ ] CKA-based 特徴量アライメントの追加
├── [ ] Non-IID 設定への対応
├── [ ] ハイパーパラメータチューニング
├── [ ] 通信効率の最適化
└── [ ] 論文用実験の実施
```

### 10.3 ディレクトリ構造

```
afad/
├── config/
│   ├── afad_config.yaml
│   └── experiments/
│       ├── exp01_fedavg.yaml
│       ├── exp02_heterofl.yaml
│       └── exp04_afad.yaml
├── src/
│   ├── __init__.py
│   ├── server/
│   │   ├── __init__.py
│   │   ├── afad_server.py
│   │   ├── strategy/
│   │   │   ├── __init__.py
│   │   │   ├── afad_strategy.py
│   │   │   ├── heterofl_aggregator.py
│   │   │   └── fedgen_distiller.py
│   │   └── generator/
│   │       ├── __init__.py
│   │       └── synthetic_generator.py
│   ├── client/
│   │   ├── __init__.py
│   │   └── afad_client.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── registry.py
│   │   ├── cnn/
│   │   │   ├── resnet.py
│   │   │   └── mobilenet.py
│   │   └── vit/
│   │       ├── vit.py
│   │       └── deit.py
│   ├── routing/
│   │   ├── __init__.py
│   │   └── family_router.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── mnist_loader.py
│   │   └── medmnist_loader.py
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py
│       ├── communication.py
│       └── visualization.py
├── tests/
│   ├── test_family_router.py
│   ├── test_generator.py
│   ├── test_aggregation.py
│   └── test_integration.py
├── scripts/
│   ├── run_server.py
│   ├── run_client.py
│   └── run_experiment.py
├── results/
│   └── .gitkeep
├── requirements.txt
├── setup.py
└── README.md
```

---

## 付録A: 参考文献

1. Li, D., & Wang, J. (2019). FedMD: Heterogenous Federated Learning via Model Distillation. NeurIPS Workshop.
2. Diao, E., et al. (2021). HeteroFL: Computation and Communication Efficient Federated Learning for Heterogeneous Clients. ICLR.
3. Zhu, Z., et al. (2021). Data-Free Knowledge Distillation for Heterogeneous Federated Learning. ICML.
4. Zhang, L., et al. (2022). Fine-tuning Global Model via Data-Free Knowledge Distillation for Non-IID Federated Learning. CVPR.
5. Zhang, J., et al. (2023). DFRD: Data-Free Robustness Distillation for Heterogeneous Federated Learning. NeurIPS.

---

## 付録B: 更新履歴

| バージョン | 日付 | 変更内容 |
|-----------|------|----------|
| 1.0 | 2025-01-16 | 初版作成 |

---

**End of Document**
