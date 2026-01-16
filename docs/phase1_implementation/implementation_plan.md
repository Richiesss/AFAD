# AFAD Implementation Plan (Phase 1)

AFAD (Adaptive Federated Architecture Distribution) の Phase 1 実装計画です。

## User Review Required

> [!IMPORTANT]
> - Phase 1ではデータセットとしてMNISTを使用します。
> - クライアントは5台の仮想クライアントとして実装し、FlowerのSimulation機能 (`flwr.simulation`) を使用して単一マシン上で動作確認を行います（GPU要件はユーザー環境に依存）。
> - 仮想環境 `venv` を `/root/workspace/afad/venv` に作成します。

## Proposed Changes

### Environment & Configuration
#### [NEW] [requirements.txt](file:///root/workspace/afad/requirements.txt)
- Torch, Flower, Torchvision, etc.

#### [NEW] [afad_config.yaml](file:///root/workspace/afad/config/afad_config.yaml)
- 実験設定、モデル設定、学習パラメータ。

### Core Components (`src/`)

#### Models (`src/models/`)
- **Registry**: モデル名からクラスをインスタンス化するファクトリー。
- **CNN**: ResNet, MobileNetの実装 (Torchvision wrapper or custom).
- **ViT**: ViT, DeiTの実装.

#### Routing (`src/routing/`)
- **FamilyRouter**: モデル構造解析、ファミリー判定、複雑度計算。

#### Server (`src/server/`)
- **AFADStrategy**: Custom Flower Strategy.
    - `configure_fit`: Generatorデータ配布、HeteroFLパラメータ抽出。
    - `aggregate_fit`: HeteroFL集約、FedGen (蒸留)。
- **SyntheticGenerator**: 潜在空間からデータを生成するGenerator。

#### Client (`src/client/`)
- **AFADClient**: Flower Client。
    - ローカル学習、KD損失の計算、Logitsの返送。

### Data (`src/data/`)
- MNISTデータローダー（IID分割）。

## Verification Plan

### Automated Tests
- **Unit Tests**:
    - `pytest tests/test_family_router.py`: ファミリー検出の正確性テスト。
    - `pytest tests/test_generator.py`: Generatorの出力形状、Backwardパスの確認。
    - `pytest tests/test_aggregation.py`: 形状の異なる重みの集約ロジック確認。

### Manual Verification
- **Simulation Run**:
    - `python scripts/run_experiment.py`: 5クライアントでの学習ループがエラーなく回り、ラウンドが進むことを確認。
    - ログ出力でAccuracyが上昇することを確認。
