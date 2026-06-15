# app.py - 광고비 기반 매출 예측 프로그램 (Streamlit)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

# ── 페이지 설정 ──────────────────────────────────────
st.set_page_config(page_title="매출 예측 프로그램", page_icon="📈", layout="centered")

# ── 제목 ─────────────────────────────────────────────
st.title("📈 매출 예측 프로그램")
st.write("광고비를 입력하면 예상 매출을 예측합니다.")
st.divider()

# ── 데이터 로드 및 모델 학습 ─────────────────────────
@st.cache_resource
def load_model():
    df = pd.read_csv('Advertising.csv', index_col=0)
    df = df.dropna()

    # IQR 이상치 제거
    Q1 = df['Sales'].quantile(0.25)
    Q3 = df['Sales'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['Sales'] >= Q1 - 1.5 * IQR) & (df['Sales'] <= Q3 + 1.5 * IQR)]

    X = df[['TV', 'Radio', 'Newspaper']].values
    y = df['Sales'].values

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    model = LinearRegression()
    model.fit(X_scaled, y_scaled)

    return model, scaler_X, scaler_y

model, scaler_X, scaler_y = load_model()

# ── 입력 UI ──────────────────────────────────────────
st.subheader("📝 광고비 입력")

col1, col2 = st.columns(2)

with col1:
    tv = st.number_input(
        "TV 광고비 (천 달러)",
        min_value=0.0, max_value=500.0,
        value=0.0, step=0.1, format="%.1f"
    )
    newspaper = st.number_input(
        "신문 광고비 (천 달러)",
        min_value=0.0, max_value=200.0,
        value=0.0, step=0.1, format="%.1f"
    )

with col2:
    radio = st.number_input(
        "라디오 광고비 (천 달러)",
        min_value=0.0, max_value=100.0,
        value=0.0, step=0.1, format="%.1f"
    )
    total = tv + radio + newspaper
    st.metric(label="총 광고비 (천 달러)", value=f"${total:.1f}천")

st.divider()

# ── 예측 버튼 ─────────────────────────────────────────
if st.button("🔍 매출 예측하기", use_container_width=True):
    new_data = pd.DataFrame({
        'TV':        [tv],
        'Radio':     [radio],
        'Newspaper': [newspaper]
    })

    new_scaled  = scaler_X.transform(new_data)
    pred_scaled = model.predict(new_scaled)
    pred_sales  = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]

    st.divider()
    st.subheader("📊 예측 결과")

    col3, col4, col5 = st.columns(3)
    col3.metric("TV 광고비",   f"${tv:.1f}천")
    col4.metric("라디오 광고비", f"${radio:.1f}천")
    col5.metric("신문 광고비",  f"${newspaper:.1f}천")

    st.success(f"📈 예측 매출: **{pred_sales:.2f}천 달러**")

    # 광고 기여도 시각화
    st.subheader("📉 광고 채널별 기여도")
    coef = model.coef_
    contrib = {
        'TV 광고':    abs(coef[0]),
        '라디오 광고': abs(coef[1]),
        '신문 광고':  abs(coef[2])
    }
    contrib_df = pd.DataFrame(list(contrib.items()), columns=['채널', '기여도'])
    contrib_df['기여도'] = contrib_df['기여도'] / contrib_df['기여도'].sum() * 100
    contrib_df = contrib_df.set_index('채널')
    st.bar_chart(contrib_df)

    st.info(f"💡 총 광고비 **${total:.1f}천** 투자 시 예측 매출은 **{pred_sales:.2f}천 달러**입니다.")