import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.utils as utils

def clamp_activation(tensor, clip_value=1e5):
    return torch.clamp(tensor, min=-clip_value, max=clip_value)

###########################################
# NaN 검사 유틸
###########################################
def check_nan_inf(tensor, label=""):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"[NaN/Inf DETECTED] {label} shape={tuple(tensor.shape)}")
        return True
    return False

class TemporalGCNLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels,
                 do_activation_clip=False,
                 activation_clip_value=1e5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.do_activation_clip = do_activation_clip
        self.activation_clip_value = activation_clip_value

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        # --- conv1 ---
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        # NaN/Inf => nan_to_num
        x = torch.nan_to_num(x,
                             nan=0.0,
                             posinf=self.activation_clip_value,
                             neginf=-self.activation_clip_value)
        if check_nan_inf(x, "GCN conv1 out (after nan_to_num)"):
            print("  conv1 out snippet:", x[:5])

        # clamp + relu
        if self.do_activation_clip:
            x = clamp_activation(x, self.activation_clip_value)
        x = torch.relu(x)

        # --- conv2 ---
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = torch.nan_to_num(x,
                             nan=0.0,
                             posinf=self.activation_clip_value,
                             neginf=-self.activation_clip_value)
        if check_nan_inf(x, "GCN conv2 out (after nan_to_num)"):
            print("  conv2 out snippet:", x[:5])

        if self.do_activation_clip:
            x = clamp_activation(x, self.activation_clip_value)
        x = torch.relu(x)

        # global_mean_pool
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        graph_emb = global_mean_pool(x, batch)
        if graph_emb.size(0) == 1:
            graph_emb = graph_emb.squeeze(0)

        graph_emb = torch.nan_to_num(graph_emb,
                                     nan=0.0,
                                     posinf=self.activation_clip_value,
                                     neginf=-self.activation_clip_value)
        return graph_emb

class TGCNCholeskyModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_nodes, seq_len=3,
                 do_activation_clip=False, activation_clip_value=1e5):
        super().__init__()
        self.seq_len = seq_len
        self.num_nodes = num_nodes

        self.gcn_layer = TemporalGCNLayer(
            in_channels, hidden_channels,
            do_activation_clip=do_activation_clip,
            activation_clip_value=activation_clip_value
        )
        self.lstm = nn.LSTM(input_size=hidden_channels,
                            hidden_size=hidden_channels,
                            num_layers=1,
                            batch_first=True)

        self.num_cholesky_elems = num_nodes * (num_nodes + 1) // 2
        self.fc_out = nn.Linear(hidden_channels, self.num_cholesky_elems)

        self.do_activation_clip = do_activation_clip
        self.activation_clip_value = activation_clip_value

    def forward(self, list_of_graph_data):
        gcn_embeddings = []
        for idx, data in enumerate(list_of_graph_data):
            emb = self.gcn_layer(
                data.x,
                data.edge_index,
                data.edge_weight,
                getattr(data, 'batch', None)
            )
            # NaN/Inf => nan_to_num
            emb = torch.nan_to_num(emb,
                                   nan=0.0,
                                   posinf=self.activation_clip_value,
                                   neginf=-self.activation_clip_value)
            if self.do_activation_clip:
                emb = clamp_activation(emb, self.activation_clip_value)
            gcn_embeddings.append(emb)

        gcn_embeddings = torch.stack(gcn_embeddings, dim=0)  # [seq_len, hidden_dim]
        gcn_embeddings = torch.nan_to_num(gcn_embeddings,
                                          nan=0.0,
                                          posinf=self.activation_clip_value,
                                          neginf=-self.activation_clip_value)
        gcn_embeddings = gcn_embeddings.unsqueeze(0)  # [1, seq_len, hidden_dim]

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(gcn_embeddings)
        lstm_out = torch.nan_to_num(lstm_out,
                                    nan=0.0,
                                    posinf=self.activation_clip_value,
                                    neginf=-self.activation_clip_value)

        final_h = lstm_out[:, -1, :]
        final_h = torch.nan_to_num(final_h,
                                   nan=0.0,
                                   posinf=self.activation_clip_value,
                                   neginf=-self.activation_clip_value)

        cholesky_vec = self.fc_out(final_h)
        cholesky_vec = torch.nan_to_num(cholesky_vec,
                                        nan=0.0,
                                        posinf=self.activation_clip_value,
                                        neginf=-self.activation_clip_value)

        # Cholesky
        L = torch.zeros(self.num_nodes, self.num_nodes, device=cholesky_vec.device)
        idx = 0
        for i in range(self.num_nodes):
            for j in range(i + 1):
                L[i, j] = cholesky_vec[0, idx]
                idx += 1
        pred_adj = L @ L.t()
        pred_adj = torch.nan_to_num(pred_adj,
                                    nan=0.0,
                                    posinf=self.activation_clip_value,
                                    neginf=-self.activation_clip_value)
        return pred_adj

class TGCNCholeskyEngine:
    def __init__(self,
                 in_channels, hidden_channels, num_nodes,
                 seq_len=3, lr=1e-3,
                 do_activation_clip=False,
                 activation_clip_value=1e5,
                 do_gradient_clip=False,
                 gradient_clip_value=1.0):

        self.model = TGCNCholeskyModel(
            in_channels, hidden_channels, num_nodes, seq_len,
            do_activation_clip=do_activation_clip,
            activation_clip_value=activation_clip_value
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.seq_len = seq_len
        self.num_nodes = num_nodes
        self.do_activation_clip = do_activation_clip
        self.activation_clip_value = activation_clip_value
        self.do_gradient_clip = do_gradient_clip
        self.gradient_clip_value = gradient_clip_value

    def train_model(self, data_sequences, epochs=10):
        """
        학습 루프:
        - batch별 loss 출력
        - epoch 완료 시점에 평균 loss 출력
        """
        for epoch in range(epochs):
            total_loss = 0.0
            valid_batch_count = 0

            for batch_idx, (list_of_data, label_adj) in enumerate(data_sequences):
                self.optimizer.zero_grad()
                pred_adj = self.model(list_of_data)

                # 추가 nan_to_num (안전장치)
                pred_adj = torch.nan_to_num(pred_adj,
                                            nan=0.0,
                                            posinf=self.activation_clip_value,
                                            neginf=-self.activation_clip_value)

                # NaN 검사
                if torch.isnan(pred_adj).any():
                    print(f"[FWD NaN] pred_adj has NaN! (epoch={epoch} batch={batch_idx}), skip.")
                    continue

                loss = self.criterion(pred_adj, label_adj)
                if torch.isnan(loss):
                    print(f"[LOSS NaN] (epoch={epoch} batch={batch_idx}), skip.")
                    continue

                loss.backward()

                if self.do_gradient_clip:
                    utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip_value)

                self.optimizer.step()

                total_loss += loss.item()
                valid_batch_count += 1

                # ---- (1) Batch Loss 출력 ----
                print(f"[Epoch {epoch+1}/{epochs}] Batch {batch_idx+1}/{len(data_sequences)} "
                      f"Loss={loss.item():.6f}")

            # epoch 끝난 뒤 평균 Loss
            if valid_batch_count > 0:
                avg_loss = total_loss / valid_batch_count
            else:
                avg_loss = 0.0
            print(f"[Epoch {epoch+1}/{epochs}] AvgLoss={avg_loss:.6f}")

    def predict(self, list_of_data):
        self.model.eval()
        with torch.no_grad():
            pred_adj = self.model(list_of_data)
            pred_adj = torch.nan_to_num(pred_adj,
                                        nan=0.0,
                                        posinf=self.activation_clip_value,
                                        neginf=-self.activation_clip_value)
        return pred_adj
