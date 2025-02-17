import numpy as np
import pandas as pd
from sqlalchemy import text

import urllib.parse as up
from sqlalchemy import create_engine

import torch
from torch_geometric.data import Data
from joblib import Parallel, delayed
from tqdm import tqdm

_GOGH_USER = 'deeptrade'
_GOGH_PASSWORD = 'Elqxmfpdlem12@'
_GOGH_ADDRESS = '147.46.216.30'
_GOGH_PORT = '13306'
_GOGH_DB = 'deeptrade'
_GOGH_URL = 'mysql+pymysql://{0}:{1}@{2}:{3}/{4}?charset=utf8mb4'.format(
    _GOGH_USER, up.quote_plus(_GOGH_PASSWORD), _GOGH_ADDRESS, _GOGH_PORT, _GOGH_DB)
ENGINE = create_engine(_GOGH_URL, echo=False, pool_recycle=3600)
PRICES = 'SNP500_PRICEVOLUME_paper'
START_DATE = "2013-01-01"
VALID_DATE = "2022-10-01"
TEST_DATE = "2023-11-01"
TEST_END = "2024-12-31"

def get_prices(start_date=START_DATE, end_date=TEST_END):
    try:
        query = f"SELECT * FROM {PRICES} WHERE date > '{start_date}' AND date <= '{end_date}'"
        with ENGINE.connect() as conn:
            result = conn.execute(text(query))
            columns = result.keys()  # 컬럼명 유지
            df = pd.DataFrame(result.fetchall(), columns=columns)
        return df
    except Exception as e:
        print(f"Error: {e}")

def clean_numeric_df(df: pd.DataFrame, cols=None, fill_method="drop"):
    """
    (A) 특정 컬럼들의 inf, -inf, NaN을 처리
    (B) fill_method="drop"이면 해당 row 제거 / "zero"이면 0으로 대체 / etc.

    - cols: None이면 df.select_dtypes(include=[np.number]) 로 판단.
    - 반환: 정리된 df
    """
    if cols is None:
        # 수치형 칼럼 자동 탐색
        cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # inf -> NaN
    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan)

    if fill_method == "drop":
        # cols 중 NaN 있는 row 제거
        df = df.dropna(subset=cols)
    elif fill_method == "zero":
        df[cols] = df[cols].fillna(0)
    else:
        # 다른 방식 (mean 등)도 가능
        for c in cols:
            if df[c].isna().any():
                mean_val = df[c].mean()
                df[c] = df[c].fillna(mean_val)

    return df

def clean_numeric_df(df: pd.DataFrame, cols=None, fill_method="drop"):
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan)
    if fill_method == "drop":
        df = df.dropna(subset=cols)
    elif fill_method == "zero":
        df[cols] = df[cols].fillna(0)
    else:
        for c in cols:
            if df[c].isna().any():
                mean_val = df[c].mean()
                df[c] = df[c].fillna(mean_val)
    return df

def create_basic_features(df):
    """
    'z_open', 'z_high', 'z_low', 'z_close', 'z_adj_close' 등을 생성.
    전일 종가/adjusted_close로 나눈 수익률 형태.
    """
    # 전일 close/adj_close
    df['prev_close'] = df.groupby('symbol')['close'].shift(1)
    df['prev_adj_close'] = df.groupby('symbol')['adjusted_close'].shift(1)

    df['z_open']      = df['open']  / df['prev_close'] - 1
    df['z_high']      = df['high']  / df['prev_close'] - 1
    df['z_low']       = df['low']   / df['prev_close'] - 1
    df['z_close']     = df['close'] / df['prev_close'] - 1
    df['z_adj_close'] = df['adjusted_close'] / df['prev_adj_close'] - 1

    # 필요 시, prev_close, prev_adj_close 자체도 계속 둘지(모델에서 필요하면 유지)
    return df

def create_rolling_features(df, k_list=[5,10,15,20,25,30]):
    """
    adjusted_close 기반의 rolling mean vs. 현재 adjusted_close를 비교
    => df['z_d{k}'] = rolling_mean / adjusted_close - 1
    """
    for k in k_list:
        # (k+1) window로 rolling, 마지막 값이 현재 행이 되도록
        # => 한 칸 앞서가거나 뒤서갈 수 있으므로 원하는 방향/align 조정 필요
        rolling_mean = df.groupby('symbol')['adjusted_close'].transform(
            lambda x: x.rolling(k+1, min_periods=k+1).mean()
        )
        # 현재 row의 adjusted_close와 비교
        df[f'z_d{k}'] = rolling_mean / df['adjusted_close'] - 1

    return df

def preprocess_data(prices: pd.DataFrame) -> pd.DataFrame:
    """
    (1) median date를 갖는 종목으로 필터
    (2) clean_numeric_df -> Inf/NaN 제거
    (3) OHLCV 기반 이동평균 컬럼: ma_5..ma_30
    (4) create_basic_features -> z_open, z_close, z_adj_close 등
    (5) create_rolling_features -> z_d5..z_d30
    (6) 재차 clean_numeric_df
    """
    # -------------------------
    # 0) 초기 numeric 정리
    # -------------------------
    numeric_cols = ['open','high','low','close','adjusted_close','volume']
    prices = clean_numeric_df(prices, cols=numeric_cols, fill_method="drop")

    # -------------------------
    # 1) median date coverage 종목 찾기
    # -------------------------
    symbol_date_count = prices.groupby('symbol')['date'].nunique()
    median_coverage = int(symbol_date_count.median())
    ref_candidates = symbol_date_count[symbol_date_count == median_coverage].index
    if len(ref_candidates) == 0:
        diff_to_median = (symbol_date_count - median_coverage).abs()
        ref_symbol = diff_to_median.idxmin()
    else:
        ref_symbol = ref_candidates[0]

    ref_df = prices.loc[prices['symbol'] == ref_symbol].sort_values('date')
    ref_dates = ref_df['date'].unique()

    valid_symbols = []
    grouped = prices.groupby('symbol')
    for sym, grp in grouped:
        sym_dates = grp['date'].unique()
        if set(ref_dates).issubset(sym_dates):
            valid_symbols.append(sym)

    print(f"[전처리 전] 전체 종목 수: {prices['symbol'].nunique()}")
    print(f"[전처리 후]  유효 종목 수: {len(valid_symbols)}")

    filtered_df = prices[
        (prices['symbol'].isin(valid_symbols)) &
        (prices['date'].isin(ref_dates))
    ].copy()
    # 날짜/심볼 순 정렬
    filtered_df.sort_values(by=['date','symbol'], inplace=True)
    filtered_df.reset_index(drop=True, inplace=True)

    # -------------------------
    # 2) 기존 이동평균 ma_5..ma_30
    # -------------------------
    ma_windows = [5,10,15,20,25,30]
    filtered_df.sort_values(by=['symbol','date'], inplace=True)
    filtered_df.reset_index(drop=True, inplace=True)
    for w in ma_windows:
        col_name = f"ma_{w}"
        filtered_df[col_name] = (
            filtered_df.groupby('symbol')['close']
                       .transform(lambda x: x.rolling(w, min_periods=w).mean())
        )
    filtered_df.sort_values(by=['date','symbol'], inplace=True)
    filtered_df.reset_index(drop=True, inplace=True)

    # -------------------------
    # 3) basic z features
    # -------------------------
    filtered_df = create_basic_features(filtered_df)

    # -------------------------
    # 4) rolling z features (adjusted_close)
    # -------------------------
    filtered_df = create_rolling_features(filtered_df, k_list=ma_windows)

    # -------------------------
    # 5) 추가 NaN/Inf 정리
    # (z_*, z_d*, ma_* 등에서 NaN이 생길 수 있음)
    # -------------------------
    # 필요한 numeric 컬럼 모아서 clean
    #  - numeric_cols + ma_5..ma_30 + z_*, z_d*
    add_cols_ma = [f"ma_{w}" for w in ma_windows]
    add_cols_zd = [f"z_d{k}" for k in ma_windows]
    z_cols = ['z_open','z_high','z_low','z_close','z_adj_close']
    all_numeric = list(set(numeric_cols + add_cols_ma + add_cols_zd + z_cols))

    filtered_df = clean_numeric_df(filtered_df, cols=all_numeric, fill_method="drop")

    return filtered_df


def _compute_graph_for_date(
    i,
    window_size,
    all_dates,
    df,
    all_symbols,
    feature_cols
):
    """
    개별 날짜 인덱스(i)에 대한 상관행렬 -> PyG Data 생성 함수.
    (원본 질문에서 설명한 대로)
    """
    current_date = all_dates[i]
    window_dates = all_dates[i - window_size + 1 : i + 1]

    window_df = df[df['date'].isin(window_dates)].copy()

    pivot_close = window_df.pivot(index='symbol', columns='date', values='close')
    pivot_close = pivot_close.reindex(index=all_symbols, columns=sorted(window_dates))

    returns = pivot_close.pct_change(axis=1).iloc[:, 1:]
    corr_matrix = returns.transpose().corr()

    current_day_df = df[df['date'] == current_date].copy()
    current_day_df.sort_values(by='symbol', inplace=True)
    node_feat = current_day_df[feature_cols].values  # shape=(num_symbols, len(feature_cols))
    x = torch.tensor(node_feat, dtype=torch.float)

    num_nodes = len(all_symbols)
    edge_indices = []
    edge_weights = []
    for r in range(num_nodes):
        for c in range(num_nodes):
            if r != c:
                edge_indices.append([r, c])
                edge_weights.append(corr_matrix.iloc[r, c])

    edge_index = torch.tensor(edge_indices, dtype=torch.long).T
    edge_weight = torch.tensor(edge_weights, dtype=torch.float)

    graph_data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
    return graph_data


def check_graph_data_validity(graph_data_list):
    """
    graph_data_list 내 각 Data 객체(x, edge_weight)에 NaN/Inf가 있는지 검사.
    필요 시 제거, 또는 로그 출력 후 수정
    """
    valid_data_list = []
    for idx, data in enumerate(graph_data_list):
        # node features
        if torch.isnan(data.x).any() or torch.isinf(data.x).any():
            print(f"[Invalid Data] index={idx}, node features contain NaN/Inf -> skip")
            continue

        # edge_weight
        if data.edge_weight is not None:
            if torch.isnan(data.edge_weight).any() or torch.isinf(data.edge_weight).any():
                print(f"[Invalid Data] index={idx}, edge_weight contain NaN/Inf -> skip")
                continue

        # OK -> keep
        valid_data_list.append(data)

    return valid_data_list


def generate_graph_data_sparsify_debug(
    df: pd.DataFrame,
    feature_cols,
    window_size=20,
    top_quantile=0.9,
    cache_path="graph_data_sparsify.pth"
):
    import os
    import numpy as np
    import torch
    from torch_geometric.data import Data

    def has_nan_or_inf(tensor, name="", debug_info=""):
        """
        Utility to check NaN/Inf in a tensor or np array. Logs if found.
        Returns True if found NaN/Inf, False otherwise.
        """
        if isinstance(tensor, np.ndarray):
            if np.isnan(tensor).any() or np.isinf(tensor).any():
                print(f"[NaN/Inf DETECTED] {name} {debug_info}")
                return True
        elif torch.is_tensor(tensor):
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                print(f"[NaN/Inf DETECTED] {name} {debug_info}")
                return True
        return False

    if os.path.exists(cache_path):
        print(f"[generate_graph_data_sparsify_debug] Loading from cache: {cache_path}")
        graph_data_list = torch.load(cache_path)
        print(f"[generate_graph_data_sparsify_debug] Loaded. Graph count={len(graph_data_list)}")

        # 추가로 로드된 그래프들에 NaN 검사 (혹시라도 캐시에 NaN이 남아있다면)
        valid_data = []
        for idx, g in enumerate(graph_data_list):
            # check node features
            if has_nan_or_inf(g.x, "node_feat", f"idx={idx}"):
                continue
            # check edge_weight
            if g.edge_weight is not None:
                if has_nan_or_inf(g.edge_weight, "edge_weight", f"idx={idx}"):
                    continue
            valid_data.append(g)

        print(f"[generate_graph_data_sparsify_debug] After checking cache: valid={len(valid_data)}, skip={len(graph_data_list)-len(valid_data)}")
        graph_data_list = valid_data
        return graph_data_list

    # --------------------
    # 새로 계산
    # --------------------
    print("[generate_graph_data_sparsify_debug] No cache found. Building new data...")

    all_dates = df['date'].unique()
    all_symbols = df['symbol'].unique()
    num_nodes = len(all_symbols)

    start_idx = window_size - 1
    end_idx = len(all_dates)

    valid_graph_data_list = []
    skip_count = 0

    for i in range(start_idx, end_idx):
        window_dates = all_dates[i - window_size + 1 : i + 1]
        current_date = all_dates[i]

        window_df = df[df['date'].isin(window_dates)].copy()
        pivot_close = window_df.pivot(index='symbol', columns='date', values='close')
        pivot_close = pivot_close.reindex(index=all_symbols, columns=sorted(window_dates))

        # returns
        returns = pivot_close.pct_change(axis=1).iloc[:, 1:]
        returns_np = returns.to_numpy()
        # 상관행렬
        corr_matrix = np.corrcoef(returns_np)

        # node_feat
        current_day_df = df[df['date'] == current_date].copy()
        current_day_df.sort_values(by='symbol', inplace=True)
        node_feat = current_day_df[feature_cols].to_numpy()

        # === NaN/Inf 검사
        if has_nan_or_inf(returns_np, "returns_np", f"day={i} date={current_date}") \
           or has_nan_or_inf(corr_matrix, "corr_matrix", f"day={i} date={current_date}") \
           or has_nan_or_inf(node_feat, "node_feat", f"day={i} date={current_date}"):
            skip_count += 1
            continue

        # 스파스화
        np.fill_diagonal(corr_matrix, 0.0)
        abs_flat = np.abs(corr_matrix).ravel()
        threshold = np.quantile(abs_flat, top_quantile)
        edge_indices = []
        edge_weights = []
        for r in range(num_nodes):
            for c in range(num_nodes):
                if r != c:
                    val = corr_matrix[r,c]
                    if abs(val) >= threshold:
                        edge_indices.append([r,c])
                        edge_weights.append(val)

        if len(edge_indices) == 0:
            edge_index = torch.zeros((2,0), dtype=torch.long)
            edge_weight = torch.zeros((0,), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).T
            edge_weight = torch.tensor(edge_weights, dtype=torch.float)

        if has_nan_or_inf(edge_weight, "edge_weight", f"day={i} date={current_date}"):
            skip_count += 1
            continue

        # Data
        x = torch.tensor(node_feat, dtype=torch.float)
        graph_data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
        valid_graph_data_list.append(graph_data)

    print(f"[generate_graph_data_sparsify_debug] day range: {start_idx}~{end_idx-1}, skip_count={skip_count}, valid={len(valid_graph_data_list)}")

    # 캐시 저장
    torch.save(valid_graph_data_list, cache_path)
    print(f"[generate_graph_data_sparsify_debug] Saved cache to {cache_path}")

    return valid_graph_data_list

