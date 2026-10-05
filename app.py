import os
import time
import pandas as pd
import streamlit as st
import yfinance as yf
from google import genai
from google.genai.errors import APIError
from streamlit_autorefresh import st_autorefresh

# 頁面基本設定
st.set_page_config(page_title="即時股票監控儀表板", layout="wide")
st.title("📈 股票清單即時診斷儀表板")

# 自動刷新：每 10 秒刷新一次頁面 (10000ms)
count = st_autorefresh(interval=10000, limit=1000, key="stock_refresh")

# 預設觀察清單
DEFAULT_STOCKS = ["2330.TW", "2454.TW", "2317.TW", "0050.TW"]

# 側邊欄設定
st.sidebar.header("⚙️ 系統設定")
user_api_key = st.sidebar.text_input("輸入 Gemini API Key:", type="password")

# 股票選擇器
selected_stocks = st.sidebar.multiselect(
    "選擇或輸入要監控的股票:",
    options=[
        "2330.TW",
        "2454.TW",
        "2317.TW",
        "2382.TW",
        "3231.TW",
        "0050.TW",
        "0056.TW",
    ],
    default=DEFAULT_STOCKS,
)


def get_stock_data(tickers):
  data = []
  for ticker in tickers:
    try:
      t = yf.Ticker(ticker)
      df = t.history(period="2d")
      if not df.empty:
        current_price = df["Close"].iloc[-1]
        prev_close = df["Close"].iloc[-2] if len(df) > 1 else current_price
        change = current_price - prev_close
        pct_change = (change / prev_close) * 100
        volume = df["Volume"].iloc[-1]

        data.append({
            "股票代號": ticker,
            "最新現價": round(current_price, 2),
            "漲跌": round(change, 2),
            "漲跌幅 (%)": round(pct_change, 2),
            "成交量": int(volume),
        })
    except Exception as e:
      st.error(f"抓取 {ticker} 失敗: {e}")
  return pd.DataFrame(data)


# 顯示行情
st.subheader(f"🔄 即時行情 (自動更新次數: {count})")
if selected_stocks:
  stock_df = get_stock_data(selected_stocks)

  # 卡片顯示
  cols = st.columns(len(selected_stocks))
  for idx, row in stock_df.iterrows():
    with cols[idx % len(cols)]:
      st.metric(
          label=row["股票代號"],
          value=f"${row['最新現價']}",
          delta=f"{row['漲跌']} ({row['漲跌幅 (%)']}%)",
      )

  # 表格顯示
  st.dataframe(stock_df, use_container_width=True)

  # AI 分析區塊
  st.divider()
  st.subheader("🤖 Gemini 股票即時診斷")

  target_stock = st.selectbox("選擇股票進行 AI 診斷:", selected_stocks)

  if st.button("🚀 產生 AI 分析報告"):
    # 讀取 API Key
    api_key = user_api_key.strip() if user_api_key else None
    if not api_key:
      try:
        api_key = st.secrets.get("GEMINI_API_KEY")
      except Exception:
        api_key = None

    if not api_key:
      st.error("❌ 找不到 API Key！請在左側欄位貼上你的 Gemini API Key。")
    else:
      with st.spinner(f"正在分析 {target_stock} (若伺服器繁忙會自動重試)..."):
        try:
          hist = yf.Ticker(target_stock).history(period="1mo")
          client = genai.Client(api_key=api_key)

          prompt = f"""
                    你是一位資深的台股籌碼與技術分析師。
                    請根據以下 {target_stock} 最近 1 個月的數據：
                    {hist.tail(10).to_string()}
                    
                    請提供 300 字內的即時分析重點：
                    1. 當前 K 線型態與均線趨勢（多/空/盤整）
                    2. 近期成交量變化的意義
                    3. 操作風險提示與關鍵支撐/壓力位
                    """

          # 優先使用 3.8 模型，並設定備用模型清單
          models_to_try = [
              "gemini-3.8-flash",
              "gemini-3.1-flash",
              "gemini-2.5-flash",
          ]
          response_text = None
          last_error = None

          for model_name in models_to_try:
            # 針對 503 等暫時性錯誤，最多重試 3 次
            for attempt in range(3):
              try:
                res = client.models.generate_content(
                    model=model_name, contents=prompt
                )
                response_text = res.text
                break
              except APIError as e:
                last_error = e
                # 若遇到 503 (UNAVAILABLE) 則等待 2 秒再重試
                if "503" in str(e) or "UNAVAILABLE" in str(e):
                  time.sleep(2)
                  continue
                else:
                  # 若遇到其他錯誤（如 404）則切換下一個模型嘗試
                  break

            if response_text:
              break

          if response_text:
            st.success("✅ 分析報告產生成功！")
            st.markdown(response_text)
          else:
            st.error(
                f"⚠️ 產出報告失敗：伺服器目前繁忙或模型暫時無法回應 ({last_error})"
            )

        except Exception as err:
          st.error(f"⚠️ 執行時發生未知錯誤：{err}")
