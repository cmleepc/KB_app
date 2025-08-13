
from pathlib import Path
from datetime import date, timedelta
from textwrap import dedent
import pandas as pd
import streamlit as st

from kb_network import (
    load_universe_kosdaq,
    select_by_sector,
    fetch_prices,
    load_external_similarity_from_df,
    build_graph_combined,
    recommend_by_graph,
)

# ---------- 고정 파라미터 ----------
ALLOWED_SECTORS = {"반도체", "금융", "소프트웨어", "제약"}
ALPHA = 0.55
SIM_THRESHOLD = 0.60
CENTRALITY_METRIC = "pagerank"
CENTRALITY_BLEND = 0.20
CENTRALITY_PROTECT_TOP_K = 1
USE_FIEDLER_TUNING = True
FIEDLER_LOW_THRESH = 0.020
FIEDLER_HOLD_LOW_THRESH = 0.010

# 기간: 최근 6개월
_today = date.today()
_start = _today - timedelta(days=180)
START_DATE = _start.strftime("%Y%m%d")
END_DATE   = _today.strftime("%Y%m%d")

# 데이터 파일
BASE_DIR = Path(__file__).parent
KOSDAQ_XLSX = BASE_DIR / "업종분류_KOSDAQ.xlsx"
SIM_CSV     = BASE_DIR / "키워드_유사도.csv"

# ---------- 공통 UI ----------
st.set_page_config(page_title="그래프 기반 네트워크 분석", layout="wide")
if "page" not in st.session_state:
    st.session_state.page = "cover"

# ---------- 표지 ----------

def render_cover():
    st.markdown(dedent("""
        <style>
        .cover-wrap { text-align:center; padding-top:10vh; font-family:'Noto Sans KR', sans-serif; }
        .cover-emoji { font-size:64px; line-height:1; margin-bottom:10px; }
        .cover-title { font-size:40px; font-weight:800; margin-bottom:6px; letter-spacing:-0.3px; }
        .cover-sub { font-size:18px; color:#5c606b; margin-bottom:28px; }
        .cover-section-title { margin-top:24px; font-weight:700; font-size:20px; }
        .cover-bullets {
            list-style:disc;
            margin:16px auto 0;
            width:min(720px,92%);
            text-align:left;
            padding-left: 100px; /* 오른쪽 들여쓰기 */
            color:#3b3f49;
            font-size:16px;
            line-height:1.7;
        }
        /* 버튼 스타일 */
        .stButton>button {
            width:220px;
            height:46px;
            font-size:16px;
            font-weight:700;
            background:#ff4b4b;
            color:#fff;
            border:0;
            border-radius:12px;
            margin-top: 40px; /* 버튼을 더 아래로 */
        }
        </style>

        <div class="cover-wrap">
          <div class="cover-emoji">📈</div>
          <div class="cover-title">투자 포트폴리오 최적화 서비스 ‘KUBO’</div>
          <div class="cover-sub">제7회 Future Finance A.I. Challenge | 팀명: 쿠보(KuBo)</div>

          <div class="cover-section-title">무엇을 하나요?</div>
          <ul class="cover-bullets">
            <li><b>주가 그래프</b> + <b>뉴스 키워드 그래프</b>를 결합해 종목 간 연결성을 계산합니다.</li>
            <li>보유 종목과 <b>연결이 강한 편입 후보</b>와, <b>연결이 약한 편출 후보</b>를 제안합니다.</li>
            <li>내부적으로 <b>중심성·Fiedler(연결 안정성)</b> 기준을 적용해 과도한 쏠림/고립을 방지합니다.</li>
          </ul>
        </div>
    """), unsafe_allow_html=True)

    # 버튼 가운데 정렬
    c1, c2, c3 = st.columns([1, 0.6, 1])
    with c2:
        if st.button("시작하기", key="start", use_container_width=True):
            st.session_state.page = "analysis"
            st.rerun()




# ---------- 이유(숫자 포함) 포맷터 ----------
def _format_reason_with_numbers(text: str, add: bool = True) -> str:
    prefix = "보유 종목과의 연결이 강함" if add else "보유 종목들과의 연결이 약함"
    if not isinstance(text, str) or not text:
        return prefix
    pairs = []
    for t in text.split(","):
        t = t.strip()
        if ":" in t:
            n, v = t.split(":", 1)
            try:
                pairs.append(f"{n.strip()}({float(v):.2f})")
            except Exception:
                pairs.append(n.strip())
        elif t:
            pairs.append(t)
    return f"{prefix} (상위 연결: {', '.join(pairs[:3])})" if pairs else prefix

# ========================= 라우팅 =========================
if st.session_state.page == "cover":
    render_cover()
    st.stop()

# ========================= 분석 페이지 ====================
st.title("그래프 기반 네트워크 분석")

# 유니버스 로드
if not KOSDAQ_XLSX.exists():
    st.error(f"업종분류 엑셀을 찾을 수 없습니다: {KOSDAQ_XLSX.name} (app.py와 같은 폴더)")
    st.stop()
try:
    uni = load_universe_kosdaq(str(KOSDAQ_XLSX))
except Exception as e:
    st.error(f"유니버스 로딩 실패: {e}")
    st.stop()
if uni.empty or "업종명" not in uni.columns:
    st.error("'업종명' 컬럼이 없거나 유니버스가 비어 있습니다.")
    st.stop()

# 섹터 4개로 제한
all_sectors = set(uni["업종명"].dropna().astype(str).unique().tolist())
sectors = sorted(list(ALLOWED_SECTORS & all_sectors))
if not sectors:
    st.error("허용 섹터(반도체/금융/소프트웨어/제약)가 업종표에 없습니다.")
    st.stop()

with st.sidebar:
    st.header("1) 섹터 선택")
    default_idx = sectors.index("반도체") if "반도체" in sectors else 0
    sector_choice = st.selectbox("섹터 (제한됨)", sectors, index=default_idx)

# 섹터 상위 50
basket = select_by_sector(uni, sector_choice, top_n=50)
if basket.empty:
    st.error(f"섹터 '{sector_choice}'에서 종목을 찾지 못했습니다.")
    st.stop()

with st.sidebar:
    st.markdown("---")
    st.header("2) 보유 종목 & 추천 개수")
    holdings = st.multiselect("보유 종목 선택", basket["종목명"].tolist(),
                              default=basket["종목명"].tolist()[:3])
    top_n = st.number_input("추천 개수(편입/편출 각각)", min_value=1, max_value=5, value=5, step=1)
    run_btn = st.button("분석 실행")

st.subheader("섹터 상위 50 (시가총액 기준)")
st.dataframe(
    basket[["종목코드", "종목명", "업종명", "시가총액"]].reset_index(drop=True),
    use_container_width=True,
)

# 외부 유사도 로드(행렬: 첫 열=종목명)
if not SIM_CSV.exists():
    st.error(f"외부 유사도 CSV를 찾을 수 없습니다: {SIM_CSV.name}")
    st.stop()
try:
    df_csv = pd.read_csv(SIM_CSV)
    ext_edges = load_external_similarity_from_df(df_csv, matrix_index_col="종목명")
except Exception as e:
    st.error(f"외부 유사도 CSV 로딩 실패: {e}")
    st.stop()

if run_btn:
    with st.spinner("가격 수집 중... (최근 6개월 자동)"):
        prices = fetch_prices(basket, START_DATE, END_DATE)
    if prices.empty:
        st.error("가격 데이터가 비었습니다.")
        st.stop()

    with st.spinner("그래프 생성 중..."):
        G = build_graph_combined(prices, ext_edges, alpha=ALPHA, sim_threshold=SIM_THRESHOLD)

    if not holdings:
        st.error("보유 종목을 1개 이상 선택하세요.")
        st.stop()

    with st.spinner("추천 산출 중..."):
        add_df, remove_df, cent_df, meta = recommend_by_graph(
            G, holdings, top_n=top_n,
            min_links_for_add=1, t_remove=0.45, remove_ratio=1/3,
            centrality_metric=CENTRALITY_METRIC,
            centrality_blend=CENTRALITY_BLEND,
            centrality_protect_top_k=CENTRALITY_PROTECT_TOP_K,
            use_fiedler_tuning=USE_FIEDLER_TUNING,
            fiedler_low_thresh=FIEDLER_LOW_THRESH,
            fiedler_hold_low_thresh=FIEDLER_HOLD_LOW_THRESH,
        )

    # 결과표: 종목명 + 이유(숫자 포함)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### ✅ 편입 추천")
        if add_df.empty:
            st.write("추천 없음")
        else:
            reason_col = next((c for c in add_df.columns if "근거" in c), None)
            show = pd.DataFrame({
                "종목명": add_df["종목명"],
                "이유": add_df[reason_col].map(lambda x: _format_reason_with_numbers(x, add=True)) if reason_col else "보유 종목과의 연결이 강함"
            })
            st.dataframe(show, use_container_width=True)

    with c2:
        st.markdown("### 🚫 편출 추천")
        if remove_df.empty:
            st.write("추천 없음")
        else:
            reason_col = next((c for c in remove_df.columns if "근거" in c), None)
            show = pd.DataFrame({
                "종목명": remove_df["종목명"],
                "이유": remove_df[reason_col].map(lambda x: _format_reason_with_numbers(x, add=False)) if reason_col else "보유 종목들과의 연결이 약함"
            })
            st.dataframe(show, use_container_width=True)

    st.success("완료!")
else:
    st.info("좌측에서 섹터/보유 종목/추천 개수를 설정하고 **분석 실행**을 눌러주세요!")


