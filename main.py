import torch
from engine import TGCNCholeskyEngine
from preprocess import preprocess_data, generate_graph_data_sparsify_debug, get_prices

##################################
# 디버깅용 함수 (NaN/Inf 검사)
##################################
def check_nan_inf(tensor, label=""):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"[NaN/Inf DETECTED] {label}, shape={tuple(tensor.shape)}")
        return True
    return False

def main():
    prices = get_prices()
    df = preprocess_data(prices)

    graph_data_list = generate_graph_data_sparsify_debug(
        df,
        feature_cols=[
            "z_open","z_high","z_low","z_close","z_adj_close",
            "ma_5","ma_10","ma_15","ma_20","ma_25","ma_30"
        ],
        window_size=20,
        top_quantile=0.9,
        cache_path="test5.pth"
    )

    num_nodes = df['symbol'].nunique()
    in_channels = 11
    seq_len = 20
    T = len(graph_data_list)
    print(f"[INFO] Generated {T} graphs.")

    # --- (A) 그래프 자체 NaN 검사 ---
    for i, gdata in enumerate(graph_data_list):
        if check_nan_inf(gdata.x, f"graph_data_list[{i}].x"):
            print("  node feature =>", gdata.x[:10])  # 예: 앞부분만
        if gdata.edge_weight is not None:
            if check_nan_inf(gdata.edge_weight, f"graph_data_list[{i}].edge_weight"):
                print("  edge_weight =>", gdata.edge_weight[:10])

    def data_to_adj(data_obj, n):
        mat = torch.zeros(n, n)
        ei = data_obj.edge_index
        ew = data_obj.edge_weight
        for k in range(ei.shape[1]):
            i = ei[0,k].item()
            j = ei[1,k].item()
            mat[i,j] = ew[k].item()
        return mat

    data_sequences = []
    for t in range(T - seq_len):
        list_of_data = graph_data_list[t : t+seq_len]
        label_index = t + seq_len
        if label_index < T:
            label_adj = data_to_adj(graph_data_list[label_index], num_nodes)
            # 라벨 NaN 검사
            if check_nan_inf(label_adj, f"label_adj (t={t})"):
                print("  label_adj =>", label_adj[:5,:5])  # 일부만
            data_sequences.append((list_of_data, label_adj))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("[INFO] device =", device)

    engine = TGCNCholeskyEngine(
        in_channels=in_channels,
        hidden_channels=16,
        num_nodes=num_nodes,
        seq_len=seq_len,
        lr=1e-3,
        do_activation_clip=True,
        activation_clip_value=1e5,
        do_gradient_clip=True,
        gradient_clip_value=5.0
    )
    engine.model.to(device)

    new_data_sequences = []
    for idx, (list_of_data, label_adj) in enumerate(data_sequences):
        gpu_list = [g.to(device) for g in list_of_data]
        label_adj = label_adj.to(device)
        # (원하면 여기서도 check_nan_inf(label_adj,"label_adj after to device") 가능)
        new_data_sequences.append((gpu_list, label_adj))

    print("[INFO] Start training ...")
    engine.train_model(new_data_sequences, epochs=5)
    print("[INFO] Done training.")

    # --- 예측 ---
    test_input = graph_data_list[-seq_len:]
    test_input_gpu = [g.to(device) for g in test_input]
    pred_adj = engine.predict(test_input_gpu)
    print("[INFO] Prediction shape:", pred_adj.shape)
    print(pred_adj)

if __name__ == "__main__":
    main()
