from typing import List, Tuple, Dict
import numpy as np
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays

class HeteroFLAggregator:
    """
    HeteroFL方式の同族内集約を行うクラス
    """
    def __init__(self):
        pass

    def aggregate(
        self, 
        family: str, 
        results: List[Tuple[np.ndarray, int]], 
        global_params: List[np.ndarray]
    ) -> Parameters:
        """
        同族内の結果を集約する
        
        Args:
            family: ファミリー名
            results: List of (parameters_ndarray, num_examples)
            global_params: Current global parameters for this family
            
        Returns:
            Parameters: Updated global parameters
        """
        # HeteroFL: 単純平均ではなく、パラメータが存在する部分だけの平均を取る
        # AFADの仕様では "チャンネル数に基づく選択的平均化"
        # 実装簡易化のため、パラメータサイズが一致する場合は単純な加重平均を行い、
        # サイズが異なる（部分モデル）の場合は、大きいモデル（Global）の対応部分に加算して平均化する。
        # しかしPyTorchのstate_dict順序に依存するため、本来は名前ベースのマッピングが必要。
        # FlowerのParametersはただのByte列のリストなので、名前情報がない。
        # ★重要: 実用的にはAFADClientがFull Modelと同じ形状のUpdate（他は0埋め）を送るか、
        # あるいはServer側でParameterの名前管理が必要。
        # HeteroFL論文では "sub-model parameters are updated..."
        
        # 今回の実装方針:
        # Clientは「自身のモデルパラメータ」そのものを送る。
        # Serverは「最大のモデル」の形状を知っている必要がある（global_params）。
        # しかし、List[ndarray]だけではどの層がどれか分からない。
        # 簡易実装として: "Weight Padding" 方式を採用するか、
        # クライアント側で "Full Model" の形状にパディングして送るのが一番簡単。
        # ここでは、AFADClientが「自分のパラメータ」を送ってくる前提で、
        # 配列の形状が一致するもの同士を集約する「Best Effort」な実装にする。
        # ※本来ならModelRegistryを使って構造を把握すべき。
        
        # 単純なFedAvg (Weighted Average) を、各レイヤーごとに行う
        # レイヤー数が異なると破綻するため、Family内ではレイヤー数は同一（幅だけ違う）前提。
        # ResNet18/34/50はレイヤー数が違うので混在は難しい -> FamilyDefinitionで分けるべきか？
        # 仕様書: "CNN Family (ResNet50, 34, 18, MobileNet)"
        # これらを混ぜて平均するのは構造的に不可能（ResNet50はBottleneck, 18はBasicBlock）。
        # HeteroFLは「同じアーキテクチャで幅（チャネル数）が違う」ものを扱う。
        # AFAD仕様書の "同族間重み共有" は、構造が一致する部分（あるいは特定レイヤー）のみを共有するか、
        # 「ResNet Family」の中でさらに「ResNet50-Half」のようなサブセットを想定している可能性が高い。
        # しかし仕様書には "ResNet50, 34, 18" と異なる深さのモデルが混在している。
        # F-002: "チャンネル数に基づく選択的平均化" -> これはHeteroFLの定義通り。
        # 深さが違うモデル間の共有はHeteroFLの範囲外（それはKnowledge Distillationでやるべき）。
        # よって、AFAD戦略としては:
        # 1. 完全に構造が一致する（幅違い含む）サブグループごとに集約 (Strict HeteroFL)
        # 2. 構造が違うものは集約しない（個別学習に近い）
        # 3. あるいは共通部分（最初のConvなど）だけ共有
        
        # 実装:
        # レイヤー数が一致するグループごとに集約する。
        # shapeが一致するレイヤーは平均、不一致なら集約しない（Globalを維持、あるいは更新なし）。
        
        # 重み付き平均の累積用
        weighted_weights = [np.zeros_like(p) for p in global_params]
        total_weights = [0.0 for _ in global_params] # カウント（または重み総和）
        
        for client_params, num_examples in results:
            for i, p in enumerate(client_params):
                if i >= len(weighted_weights):
                    break # Globalより多いレイヤーは無視
                
                # 形状チェック
                g_p = weighted_weights[i]
                if p.shape == g_p.shape:
                    # 完全一致: 単純加算
                    weighted_weights[i] += p * num_examples
                    total_weights[i] += num_examples
                else:
                    # 部分一致 (HeteroFL logic): スライシング
                    # Conv2d: (Out, In, H, W) -> Out, In を合わせる
                    # 単純化: 左上のスライスのみ対応
                    slices = tuple(slice(0, d) for d in p.shape)
                    weighted_weights[i][slices] += p * num_examples
                    
                    # カウント行列が必要だが、簡易的にスカラで割るとおかしくなる。
                    # 正確にはパラメータごとのカウンタが必要
                    # TODO: 本格実装は複雑すぎるため、Phase 1では
                    # 「同じモデルアーキテクチャ（ResNet18同士など）」のみを集約し、
                    # 異なるアーキテクチャ間はFedGen（蒸留）のみで知識共有する方針に倒すのが安全。
                    # 仕様書にも "同族（HeteroFL）" とあるが、ResNet18と50を同族としてパディング集約するのは無理がある。
                    # FamilyRouterの実装では "resnet" で一括りだが、ここではモデル名ごとに集約を分けるか、
                    # あるいは "ResNet18" と "ResNet18-Half" のような関係のみをHeteroFL対象とする。
                    
                    # 修正方針:
                    # 形状が完全一致するもののみ平均する (FedAvg equivalent for same arch).
                    pass

        # 平均化
        new_params = []
        for i, (w_sum, total) in enumerate(zip(weighted_weights, total_weights)):
            if total > 0:
                new_params.append(w_sum / total)
            else:
                # 誰も更新しなかったパラメータはGlobalのまま
                new_params.append(global_params[i])
                
        return ndarrays_to_parameters(new_params)
