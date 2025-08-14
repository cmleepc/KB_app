#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# kb_network.py
# 그래프 기반 네트워크 분석 (+ 중심성 + Fiedler) — KOSDAQ 전용

from typing import Optional, List, Tuple, Union
import io
import unicodedata
import numpy as np
import pandas as pd
import networkx as nx

# KRX
from pykrx import stock


# -----------------------------
# 유틸
# -----------------------------
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s))
    return s.strip()

def _zscore(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    return (s - s.mean()) / (s.std() + 1e-8)


# -----------------------------
# 1) 유니버스 로딩 (KOSDAQ 전용)
# -----------------------------
def load_universe_kosdaq(kosdaq_xlsx: Union[str, io.BytesIO]) -> pd.DataFrame:
    """
    KOSDAQ 업종분류 엑셀에서 기본 컬럼만 추출:
    ['종목코드','종목명','업종명','시가총액']
    """
    cols = ["종목코드", "종목명", "업종명", "시가총액"]
    df = pd.read_excel(kosdaq_xlsx)
    df = df[[c for c in cols if c in df.columns]].copy()

    if "종목코드" in df.columns:
        df["종목코드"] = df["종목코드"].astype(str).str.zfill(6)
    if "종목명" in df.columns:
        df["종목명"] = df["종목명"].astype(str).map(_norm)
    if "업종명" in df.columns:
        df["업종명"] = df["업종명"].astype(str).map(_norm)
    if "시가총액" in df.columns:
        df["시가총액"] = pd.to_numeric(df["시가총액"], errors="coerce")
    return df


# -----------------------------
# 2) 섹터 선택 (유연 매칭, 시총 상위 top_n)
# -----------------------------
def _sector_columns(universe: pd.DataFrame) -> List[str]:
    candidates = ["업종명", "업종", "업종대분류", "세부업종명"]
    return [c for c in candidates if c in universe.columns]

def select_by_sector(universe: pd.DataFrame, sector: str, top_n: int = 50) -> pd.DataFrame:
    if universe.empty:
        raise ValueError("유니버스가 비어 있습니다.")
    sector_cols = _sector_columns(universe)
    if not sector_cols:
        raise ValueError(f"업종 컬럼을 찾지 못했습니다. (가용 컬럼: {list(universe.columns)})")

    q = _norm(sector)
    exact_mask = False
    for c in sector_cols:
        exact_mask = exact_mask | (universe[c].astype(str).map(_norm) == q)
    df = universe[exact_mask].copy()

    if df.empty:
        contains_mask = False
        for c in sector_cols:
            contains_mask = contains_mask | (universe[c].astype(str).map(_norm).str.contains(q, na=False))
        df = universe[contains_mask].copy()

        if df.empty:
            # UI에서 처리하기 위해 빈 DF 반환
            return pd.DataFrame(columns=universe.columns)

    if "시가총액" in df.columns:
        df["시가총액"] = pd.to_numeric(df["시가총액"], errors="coerce")
        df = df.sort_values("시가총액", ascending=False)

    return df.head(top_n)


# -----------------------------
# 3) 주가 수집 (종가)
# -----------------------------
def fetch_prices(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """
    반환: index=날짜, columns=종목명, 값=종가
    """
    result = pd.DataFrame()
    for _, row in df.iterrows():
        code, name = row["종목코드"], row["종목명"]
        try:
            ohlcv = stock.get_market_ohlcv_by_date(start_date, end_date, code)
            close = ohlcv["종가"].rename(name)
            result = pd.concat([result, close], axis=1)
        except Exception:
            # 호출측에서 UI 경고 처리 권장
            pass

    result = result.dropna(axis=1)
    if not result.empty:
        result = result.loc[:, result.std() > 0]
    return result


# -----------------------------
# 4) 가격 → 유사도(0~1)
# -----------------------------
def prices_to_similarity(prices: pd.DataFrame) -> pd.DataFrame:
    corr = prices.corr().clip(-1, 1).fillna(0.0)
    sim  = (corr + 1.0) / 2.0
    np.fill_diagonal(sim.values, 0.0)
    return sim


# -----------------------------
# 5) 외부 CSV 유사도 로더 (DF 입력)
# -----------------------------
def load_external_similarity_from_df(df: pd.DataFrame,
                                     name_col_a: Optional[str] = None,
                                     name_col_b: Optional[str] = None,
                                     sim_col: Optional[str] = None,
                                     matrix_index_col: Optional[str] = None) -> pd.DataFrame:
    """
    df: 업로드된 CSV를 읽은 DataFrame
    지원 포맷:
      1) 엣지 리스트: (A,B,sim) / (source,target,weight) 등
      2) 행렬: 첫 열=종목명, 나머지 유사도 값
    반환: DataFrame(columns=['A','B','sim']) with A!=B, sim∈[0,1]
    """
    # 엣지 리스트 감지
    a_col, b_col, s_col = name_col_a, name_col_b, sim_col
    if a_col is None or b_col is None:
        for ca, cb in [("종목A","종목B"), ("종목1","종목2"), ("A","B"), ("source","target"), ("from","to")]:
            if ca in df.columns and cb in df.columns:
                a_col, b_col = ca, cb
                break
    if s_col is None:
        for c in ["유사도","sim","similarity","weight","score"]:
            if c in df.columns:
                s_col = c
                break

    if a_col is not None and b_col is not None and s_col is not None:
        out = df[[a_col, b_col, s_col]].copy()
        out.columns = ["A","B","sim"]
        out["A"] = out["A"].map(_norm)
        out["B"] = out["B"].map(_norm)
        out = out[out["A"] != out["B"]]
        out["sim"] = pd.to_numeric(out["sim"], errors="coerce")
        out = out.dropna(subset=["sim"])
        out["sim"] = out["sim"].clip(lower=0.0)
        out[["A2","B2"]] = np.sort(out[["A","B"]].values, axis=1)
        out = (out.groupby(["A2","B2"], as_index=False)["sim"]
                 .mean()
                 .rename(columns={"A2":"A","B2":"B"}))
        out["sim"] = out["sim"].clip(0,1)
        return out

    # 행렬 감지
    idx_col = matrix_index_col
    if idx_col is None:
        idx_col = "종목명" if "종목명" in df.columns else df.columns[0]

    df2 = df.copy()
    df2.columns = [_norm(c) for c in df2.columns]
    idx_col = _norm(idx_col)
    if idx_col not in df2.columns:
        raise ValueError(f"행렬 포맷 추정 실패: 인덱스 컬럼 '{idx_col}' 없음.")

    mat = df2.set_index(idx_col)
    for c in mat.columns:
        mat[c] = pd.to_numeric(mat[c], errors="coerce")

    out = mat.stack(dropna=True).reset_index()
    out.columns = ["A","B","sim"]
    out["A"] = out["A"].map(_norm)
    out["B"] = out["B"].map(_norm)
    out = out[out["A"] != out["B"]]
    out["sim"] = out["sim"].fillna(0.0).clip(lower=0.0)

    out[["A2","B2"]] = np.sort(out[["A","B"]].values, axis=1)
    out = (out.groupby(["A2","B2"], as_index=False)["sim"]
             .mean()
             .rename(columns={"A2":"A","B2":"B"}))
    out["sim"] = out["sim"].clip(0,1)
    return out


# -----------------------------
# 6) 결합 그래프(가격+뉴스)
# -----------------------------
def build_graph_combined(prices: pd.DataFrame,
                         ext_edges: pd.DataFrame,
                         alpha: float = 0.5,
                         sim_threshold: float = 0.6) -> nx.Graph:
    sim_p = prices_to_similarity(prices)  # [0,1]

    names_norm = {_norm(n): n for n in prices.columns}
    ext = ext_edges.copy()
    ext = ext[ext["A"].isin(names_norm.keys()) & ext["B"].isin(names_norm.keys())]

    # 빠른 조회 dict
    ext_key = tuple(map(tuple, np.sort(ext[["A", "B"]].values, axis=1)))
    ext_sim = dict(zip(ext_key, ext["sim"]))

    G = nx.Graph()
    G.add_nodes_from(prices.columns)

    cols = list(prices.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a_raw, b_raw = cols[i], cols[j]
            a, b = _norm(a_raw), _norm(b_raw)

            s_p = float(sim_p.loc[a_raw, b_raw])
            s_e = float(ext_sim.get(tuple(sorted((a, b))), 0.0))

            s = alpha * s_p + (1.0 - alpha) * s_e
            if s >= sim_threshold:
                G.add_edge(a_raw, b_raw,
                           weight=max(0.0, 1.0 - s),
                           sim=float(s),
                           s_price=float(s_p),
                           s_ext=float(s_e))

    for node in prices.columns:
        if node not in G:
            G.add_node(node)

    return G


# -----------------------------
# 7) Fiedler value
# -----------------------------
def fiedler_value(G: nx.Graph, weight: str = "sim") -> float:
    if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
        return 0.0
    if not nx.is_connected(G):
        return 0.0
    L = nx.laplacian_matrix(G, weight=weight)
    # scipy sparse면 toarray(), ndarray면 그대로
    if hasattr(L, "toarray"):
        L = L.toarray()
    else:
        L = np.asarray(L)
    L = L.astype(float)

    eigvals = np.linalg.eigvalsh(L)
    eigvals = np.sort(np.real(eigvals))
    return float(eigvals[1]) if len(eigvals) >= 2 else 0.0

def fiedler_value_holdings(G: nx.Graph, holdings: List[str], weight: str = "sim") -> float:
    H = [h for h in holdings if h in G]
    if len(H) < 2:
        return 0.0
    SG = G.subgraph(H).copy()
    return fiedler_value(SG, weight=weight)


# -----------------------------
# 8) 중심성 표
# -----------------------------
def centrality_table(G: nx.Graph) -> pd.DataFrame:
    deg = nx.degree_centrality(G)
    strength = {n: sum(d.get("sim", 1.0 - d.get("weight", 1.0))
                       for _, _, d in G.edges(n, data=True)) for n in G.nodes()}
    btw = nx.betweenness_centrality(G, weight="weight", normalized=True)
    cls = nx.closeness_centrality(G, distance="weight")
    try:
        eig = nx.eigenvector_centrality_numpy(G, weight="sim")
    except Exception:
        eig = {n: 0.0 for n in G.nodes()}
    try:
        pr = nx.pagerank(G, weight="sim")
    except Exception:
        pr = {n: 0.0 for n in G.nodes()}

    df = pd.DataFrame({
        "degree": pd.Series(deg),
        "strength": pd.Series(strength),
        "betweenness": pd.Series(btw),
        "closeness": pd.Series(cls),
        "eigenvector": pd.Series(eig),
        "pagerank": pd.Series(pr),
    }).fillna(0.0)
    df.index.name = "종목명"
    return df


# -----------------------------
# 9) 엣지 sim 헬퍼
# -----------------------------
def _edge_similarity(G: nx.Graph, u: str, v: str) -> float:
    data = G.get_edge_data(u, v, default=None)
    if not data:
        return 0.0
    if "sim" in data:
        return float(data["sim"])
    w = float(data.get("weight", 1.0))
    return max(0.0, min(1.0, 1.0 - w))


# -----------------------------
# 10) 추천 (중심성 + Fiedler 튜닝)
# -----------------------------
def recommend_by_graph(G: nx.Graph,
                       holdings: List[str],
                       top_n: int = 5,
                       min_links_for_add: int = 1,
                       t_remove: float = 0.45,
                       remove_ratio: float = 1/3,
                       centrality_metric: str = "pagerank",
                       centrality_blend: float = 0.2,
                       centrality_protect_top_k: int = 1,
                       use_fiedler_tuning: bool = True,
                       fiedler_low_thresh: float = 0.02,
                       fiedler_hold_low_thresh: float = 0.01):
    nodes = set(G.nodes)
    H = [h for h in holdings if h in nodes]
    others = [n for n in nodes if n not in H]

    # Fiedler 계산 & 튜닝
    f_global = fiedler_value(G, weight="sim")
    f_hold   = fiedler_value_holdings(G, H, weight="sim")

    min_links_eff = int(min_links_for_add)
    t_remove_eff  = float(t_remove)
    centrality_blend_eff = float(centrality_blend)

    if use_fiedler_tuning:
        if f_global < fiedler_low_thresh:
            min_links_eff = max(min_links_eff + 1, 1)
            centrality_blend_eff = min(centrality_blend_eff + 0.1, 0.6)
        if f_hold < fiedler_hold_low_thresh:
            t_remove_eff = min(max(t_remove_eff, 0.48), 0.6)

    # 중심성
    cent_df = centrality_table(G)
    if centrality_metric not in cent_df.columns:
        raise ValueError(f"알 수 없는 중심성 지표 '{centrality_metric}'. 사용 가능: {list(cent_df.columns)}")
    cent = cent_df[centrality_metric].copy()
    cent_z = _zscore(cent)

    # 편입
    add_scores, add_reasons, add_links = {}, {}, {}
    for s in others:
        sims = [(h, _edge_similarity(G, s, h)) for h in H]
        sims = [(h, v) for h, v in sims if v > 0]
        if not sims:
            continue
        add_links[s] = len(sims)
        score_link = float(np.mean([v for _, v in sims]))
        add_scores[s] = score_link
        top3 = sorted(sims, key=lambda x: x[1], reverse=True)[:3]
        add_reasons[s] = ", ".join([f"{h}:{v:.2f}" for h, v in top3])

    filtered = [k for k in add_scores.keys() if add_links.get(k, 0) >= min_links_eff]
    if filtered:
        base = pd.Series({k: add_scores[k] for k in filtered})
        if centrality_blend_eff > 0:
            c = cent_z.reindex(base.index).fillna(0.0)
            combo = _zscore(base) * (1.0 - centrality_blend_eff) + c * centrality_blend_eff
        else:
            combo = base
        add_scores = combo.to_dict()

    add_rank = sorted(add_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    add_df = pd.DataFrame(add_rank, columns=["종목명", "점수"])
    if not add_df.empty:
        label_col = "평균연결도(중심성/Fiedler 튜닝)" if (use_fiedler_tuning and centrality_blend_eff > 0) \
                    else ("평균연결도(중심성블렌딩)" if centrality_blend_eff > 0 else "평균연결도")
        add_df.rename(columns={"점수": label_col}, inplace=True)
        add_df["근거(보유 상위연결 3)"] = add_df["종목명"].map(add_reasons)

    # 편출
    remove_scores, remove_reasons = {}, {}
    for h in H:
        sims = [(h2, _edge_similarity(G, h, h2)) for h2 in H if h2 != h]
        score = float(np.mean([v for _, v in sims])) if sims else 0.0
        remove_scores[h] = score
        top3 = sorted(sims, key=lambda x: x[1], reverse=True)[:3]
        remove_reasons[h] = ", ".join([f"{n}:{v:.2f}" for n, v in top3]) if sims else ""

    protect_set = set()
    if centrality_protect_top_k > 0 and len(H) > 0:
        cent_holdings = cent.reindex(H).fillna(0.0).sort_values(ascending=False)
        protect_set = set(cent_holdings.head(centrality_protect_top_k).index.tolist())

    cand = [(k, v) for k, v in remove_scores.items() if (k not in protect_set and v < t_remove_eff)]
    cand_sorted = sorted(cand, key=lambda x: x[1])
    max_remove = max(1, int(len(H) * remove_ratio))
    cand_sorted = cand_sorted[:max(0, min(top_n, max_remove))]
    remove_df = pd.DataFrame(cand_sorted, columns=["종목명", "보유내 평균연결도"])
    if not remove_df.empty:
        remove_df["근거(보유내 상위연결 3)"] = remove_df["종목명"].map(remove_reasons)

    # 메타
    meta = {
        "fiedler_value_global": f_global,
        "fiedler_value_holdings": f_hold
    }

    return add_df, remove_df, cent_df, meta

