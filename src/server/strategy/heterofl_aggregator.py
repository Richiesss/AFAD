import numpy as np
from flwr.common import Parameters, ndarrays_to_parameters


class HeteroFLAggregator:
    """
    HeteroFL方式の集約を行うクラス

    参照実装: https://github.com/diaoenmao/HeteroFL-Computation-and-Communication-Efficient-Federated-Learning-for-Heterogeneous-Clients

    主要な機能:
    1. split_model: クライアントのmodel_rateに基づいてサブモデルのインデックスを計算
    2. distribute: グローバルモデルからサブモデルを切り出してクライアントに配布
    3. aggregate: クライアントからの更新を集約（更新回数ベース）
    """

    def __init__(self, global_model_rate: float = 1.0):
        """
        Args:
            global_model_rate: グローバルモデルの基準レート（通常1.0）
        """
        self.global_model_rate = global_model_rate
        # クライアントごとのパラメータインデックスを保持
        self.client_param_idx: dict[str, list[tuple]] = {}

    def compute_param_idx(
        self, global_params: list[np.ndarray], model_rate: float
    ) -> list[tuple]:
        """
        クライアントのmodel_rateに基づいてパラメータインデックスを計算

        HeteroFLでは、小さいサブモデルは大きいモデルの先頭部分を使用する

        Args:
            global_params: グローバルモデルのパラメータリスト
            model_rate: クライアントのモデルレート（0.0〜1.0）
                       例: 0.5 = グローバルの半分の幅

        Returns:
            各レイヤーのインデックスタプルのリスト
        """
        param_idx = []
        scaler = model_rate / self.global_model_rate

        # 前のレイヤーの出力インデックス（次のレイヤーの入力に使用）
        prev_output_idx = None

        for i, param in enumerate(global_params):
            shape = param.shape

            if len(shape) == 0:
                # スカラー（例: BatchNormのnum_batches_tracked）
                param_idx.append(())
                continue

            elif len(shape) == 1:
                # 1次元（バイアス、BatchNormのweight/bias等）
                # 前のレイヤーの出力サイズに合わせる
                if prev_output_idx is not None:
                    idx = prev_output_idx
                else:
                    output_size = int(np.ceil(shape[0] * scaler))
                    idx = np.arange(output_size)
                param_idx.append((idx,))

            elif len(shape) == 2:
                # 2次元（全結合層: [out_features, in_features]）
                output_size = int(np.ceil(shape[0] * scaler))
                output_idx = np.arange(output_size)

                if prev_output_idx is not None:
                    input_idx = prev_output_idx
                else:
                    input_size = int(np.ceil(shape[1] * scaler))
                    input_idx = np.arange(input_size)

                param_idx.append((output_idx, input_idx))
                prev_output_idx = output_idx

            elif len(shape) == 4:
                # 4次元（Conv2d: [out_channels, in_channels, H, W]）
                output_size = int(np.ceil(shape[0] * scaler))
                output_idx = np.arange(output_size)

                if prev_output_idx is not None:
                    input_idx = prev_output_idx
                else:
                    input_size = int(np.ceil(shape[1] * scaler))
                    input_idx = np.arange(input_size)

                param_idx.append((output_idx, input_idx))
                prev_output_idx = output_idx

            else:
                # その他の次元
                scaled_shape = tuple(int(np.ceil(s * scaler)) for s in shape)
                idx_tuple = tuple(np.arange(s) for s in scaled_shape)
                param_idx.append(idx_tuple)
                if len(shape) > 0:
                    prev_output_idx = np.arange(scaled_shape[0])

        return param_idx

    def distribute(
        self, global_params: list[np.ndarray], client_id: str, model_rate: float
    ) -> list[np.ndarray]:
        """
        グローバルモデルからクライアント用のサブモデルを切り出す

        Args:
            global_params: グローバルモデルのパラメータリスト
            client_id: クライアントID
            model_rate: クライアントのモデルレート

        Returns:
            サブモデルのパラメータリスト
        """
        # インデックスを計算して保存
        param_idx = self.compute_param_idx(global_params, model_rate)
        self.client_param_idx[client_id] = param_idx

        # サブモデルを切り出す
        sub_params = []
        for i, (param, idx) in enumerate(zip(global_params, param_idx)):
            if len(idx) == 0:
                # スカラー
                sub_params.append(param.copy())
            elif len(idx) == 1:
                # 1次元（バイアス等）- スライスで切り出し
                size = len(idx[0])
                sub_params.append(param[:size].copy())
            elif len(idx) == 2:
                out_size = len(idx[0])
                in_size = len(idx[1])
                if len(param.shape) == 2:
                    # 2次元（全結合層: [out, in]）
                    sub_params.append(param[:out_size, :in_size].copy())
                elif len(param.shape) == 4:
                    # 4次元（Conv2d: [out, in, H, W]）
                    sub_params.append(param[:out_size, :in_size, :, :].copy())
                else:
                    sub_params.append(param.copy())
            else:
                # その他
                sub_params.append(param.copy())

        return sub_params

    def aggregate(
        self,
        family: str,
        results: list[tuple[str, np.ndarray, int]],  # (client_id, params, num_examples)
        global_params: list[np.ndarray],
    ) -> Parameters:
        """
        HeteroFLの集約ロジック

        参照実装のcombineメソッドに基づく:
        - 各位置の更新回数をカウント（サンプル数ではない）
        - 更新があった位置のみ平均化
        - 更新がなかった位置は元のグローバル値を保持

        Args:
            family: ファミリー名
            results: [(client_id, parameters_ndarray, num_examples), ...]
            global_params: 現在のグローバルパラメータ

        Returns:
            Parameters: 更新されたグローバルパラメータ
        """
        if not results:
            return ndarrays_to_parameters(global_params)

        # 累積値とカウント用のバッファを初期化
        accumulated = [np.zeros_like(p) for p in global_params]
        count = [np.zeros_like(p) for p in global_params]

        for client_id, client_params, num_examples in results:
            # クライアントのインデックスを取得
            param_idx = self.client_param_idx.get(client_id)

            if param_idx is None:
                # インデックスがない場合は形状ベースで推定（後方互換性）
                param_idx = self._infer_param_idx(client_params, global_params)

            for i, (local_param, idx) in enumerate(zip(client_params, param_idx)):
                if i >= len(accumulated):
                    break

                if len(idx) == 0:
                    # スカラー
                    accumulated[i] += local_param
                    count[i] += 1
                elif len(idx) == 1:
                    # 1次元 - スライスで処理
                    size = len(idx[0])
                    accumulated[i][:size] += local_param
                    count[i][:size] += 1
                elif len(idx) == 2:
                    out_size = len(idx[0])
                    in_size = len(idx[1])
                    if len(global_params[i].shape) == 2:
                        # 2次元（全結合層）- スライスで処理
                        accumulated[i][:out_size, :in_size] += local_param
                        count[i][:out_size, :in_size] += 1
                    elif len(global_params[i].shape) == 4:
                        # 4次元（Conv2d）- スライスで処理
                        accumulated[i][:out_size, :in_size, :, :] += local_param
                        count[i][:out_size, :in_size, :, :] += 1
                    else:
                        # フォールバック: スライスベース
                        slices = tuple(slice(0, len(ix)) for ix in idx)
                        accumulated[i][slices] += local_param
                        count[i][slices] += 1

        # 平均化
        new_params = []
        for acc, cnt, orig in zip(accumulated, count, global_params):
            updated = orig.copy()
            mask = cnt > 0
            if np.any(mask):
                updated[mask] = acc[mask] / cnt[mask]
            new_params.append(updated)

        return ndarrays_to_parameters(new_params)

    def _infer_param_idx(
        self, client_params: list[np.ndarray], global_params: list[np.ndarray]
    ) -> list[tuple]:
        """
        クライアントパラメータの形状からインデックスを推定（後方互換性用）

        Args:
            client_params: クライアントのパラメータ
            global_params: グローバルパラメータ

        Returns:
            推定されたインデックスリスト
        """
        param_idx = []
        for client_p, global_p in zip(client_params, global_params):
            if client_p.shape == global_p.shape:
                # 同じ形状 - フルモデル
                idx = tuple(np.arange(s) for s in client_p.shape)
            else:
                # 異なる形状 - サブモデル
                idx = tuple(np.arange(s) for s in client_p.shape)
            param_idx.append(idx)
        return param_idx

    def aggregate_simple(
        self,
        family: str,
        results: list[tuple[np.ndarray, int]],
        global_params: list[np.ndarray],
    ) -> Parameters:
        """
        シンプルな集約（client_idなしの後方互換性用）

        インデックス追跡なしで、形状ベースのスライスで集約

        Args:
            family: ファミリー名
            results: [(parameters_ndarray, num_examples), ...]
            global_params: 現在のグローバルパラメータ

        Returns:
            Parameters: 更新されたグローバルパラメータ
        """
        if not results:
            return ndarrays_to_parameters(global_params)

        # 累積値とカウント用のバッファを初期化
        accumulated = [np.zeros_like(p) for p in global_params]
        count = [np.zeros_like(p) for p in global_params]

        for client_params, num_examples in results:
            for i, p in enumerate(client_params):
                if i >= len(accumulated):
                    break

                target_shape = accumulated[i].shape
                src_shape = p.shape

                if len(target_shape) != len(src_shape):
                    continue

                # 各次元でサイズが収まるか確認
                valid = all(s <= t for s, t in zip(src_shape, target_shape))
                if not valid:
                    continue

                slices = tuple(slice(0, d) for d in src_shape)

                # 更新回数でカウント（サンプル数ではない）
                accumulated[i][slices] += p
                count[i][slices] += 1

        # 平均化
        new_params = []
        for acc, cnt, orig in zip(accumulated, count, global_params):
            updated = orig.copy()
            mask = cnt > 0
            if np.any(mask):
                updated[mask] = acc[mask] / cnt[mask]
            new_params.append(updated)

        return ndarrays_to_parameters(new_params)
