import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(
    page_title="Trader Behaviour Analytics",
    page_icon="📈",
    layout="wide"
)

# Load data
folder = ""

df_daily  = pd.read_csv("df_daily.csv")
df_merged = pd.read_csv("df_merged.csv")

df_daily['date']  = pd.to_datetime(df_daily['date'])
df_merged['date'] = pd.to_datetime(df_merged['date'])

# Sidebar
st.sidebar.title("📈 Navigation")
page = st.sidebar.radio("Go to", [
    "Project Overview",
    "Market Sentiment",
    "Trader Performance",
    "Trader Segments",
    "Strategy Recommendations"
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Data Summary**")
st.sidebar.metric("Total Trades", "104,272")
st.sidebar.metric("Unique Traders", "32")
st.sidebar.metric("Date Range", "Dec 2023 - May 2025")

# ── PAGE 1 — Project Overview ──────────────────
if page == "Project Overview":
    st.title("📈 Trader Behaviour & Performance Analytics")
    st.subheader("Using Crypto Market Sentiment to Analyse Trader Decisions on Hyperliquid DEX")
    st.markdown("""
    This project analyses how **Bitcoin Fear & Greed sentiment Index** influences 
    crypto trader behaviour and performance on the **Hyperliquid DEX**.
    The Fear & Greed Index reflects overall crypto market mood — 
    affecting all assets traded on the exchange.
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Trades", "104,272")
    col2.metric("Unique Traders", "32")
    col3.metric("Trader-Days", "1,690")
    col4.metric("Model Accuracy", "96%")

    st.markdown("---")
    st.subheader("Key Findings")
    col1, col2 = st.columns(2)
    with col1:
        st.success("Fear days produce highest avg PnL ($7,410)")
        st.success("Traders place 2x bigger bets during Fear")
        st.success("High Risk traders make 2.7x more on Fear days")
    with col2:
        st.info("Traders 47% more active on Greed days")
        st.info("Consistent short bias (60%) across all sentiments")
        st.info("Win rate stays ~84% regardless of sentiment")

    st.markdown("---")
    st.subheader("Project Methodology")
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown("**Step 1**\nData Preparation\n\n211K trades cleaned and merged")
    col2.markdown("**Step 2**\nEDA\n\n8 charts, 7 insights found")
    col3.markdown("**Step 3**\nSegmentation\n\n3 trader segments identified")
    col4.markdown("**Step 4**\nModelling\n\n96% accuracy Random Forest")

# ── PAGE 2 — Market Sentiment ──────────────────
elif page == "Market Sentiment":
    st.title("📊 Market Sentiment Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sentiment_counts = df_daily['sentiment_binary'].value_counts()
        colors = {'Fear': '#e74c3c', 'Neutral': '#95a5a6', 'Greed': '#2ecc71'}
        sentiment_counts.reindex(['Fear', 'Neutral', 'Greed']).plot(
            kind='bar', color=[colors[x] for x in ['Fear', 'Neutral', 'Greed']], ax=ax)
        ax.set_title('Trader-Days by Sentiment')
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Count')
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("Sentiment Value Over Time")
        fig, ax = plt.subplots(figsize=(6, 4))
        daily_sentiment = df_daily.groupby('date')['value'].mean()
        ax.plot(daily_sentiment.index, daily_sentiment.values, 
                color='#3498db', linewidth=1)
        ax.axhline(y=50, color='gray', linestyle='--', linewidth=0.8)
        ax.fill_between(daily_sentiment.index, daily_sentiment.values, 50,
                        where=(daily_sentiment.values > 50),
                        color='#2ecc71', alpha=0.3, label='Greed')
        ax.fill_between(daily_sentiment.index, daily_sentiment.values, 50,
                        where=(daily_sentiment.values < 50),
                        color='#e74c3c', alpha=0.3, label='Fear')
        ax.set_title('Fear & Greed Index Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sentiment Value')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Sentiment Statistics")
    stats = df_daily.groupby('sentiment_binary')['value'].agg(
        ['mean', 'min', 'max', 'count']).round(2)
    st.dataframe(stats, use_container_width=True)

# ── PAGE 3 — Trader Performance ────────────────
elif page == "Trader Performance":
    st.title("💰 Trader Performance by Sentiment")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Average Net PnL by Sentiment")
        fig, ax = plt.subplots(figsize=(6, 4))
        pnl = df_daily.groupby('sentiment_binary')['net_pnl'].mean().reindex(
            ['Fear', 'Neutral', 'Greed'])
        colors = ['#e74c3c', '#95a5a6', '#2ecc71']
        ax.bar(pnl.index, pnl.values, color=colors)
        ax.set_title('Average Net PnL by Sentiment')
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Average Net PnL (USD)')
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("Average Trade Count by Sentiment")
        fig, ax = plt.subplots(figsize=(6, 4))
        trades = df_daily.groupby('sentiment_binary')['trade_count'].mean().reindex(
            ['Fear', 'Neutral', 'Greed'])
        ax.bar(trades.index, trades.values, color=colors)
        ax.set_title('Avg Trades per Day by Sentiment')
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Average Trade Count')
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Average Position Size by Sentiment")
        fig, ax = plt.subplots(figsize=(6, 4))
        size = df_daily.groupby('sentiment_binary')['avg_size_usd'].mean().reindex(
            ['Fear', 'Neutral', 'Greed'])
        ax.bar(size.index, size.values, color=colors)
        ax.set_title('Avg Position Size by Sentiment')
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Average Size USD')
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("Win Rate by Sentiment")
        fig, ax = plt.subplots(figsize=(6, 4))
        wr = df_daily.groupby('sentiment_binary')['win_rate'].mean().reindex(
            ['Fear', 'Neutral', 'Greed'])
        ax.bar(wr.index, wr.values, color=colors)
        ax.set_title('Avg Win Rate by Sentiment')
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Win Rate (0 to 1)')
        ax.set_ylim(0, 1)
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Performance Summary Table")
    summary = df_daily.groupby('sentiment_binary').agg(
        avg_net_pnl    = ('net_pnl', 'mean'),
        median_net_pnl = ('net_pnl', 'median'),
        avg_win_rate   = ('win_rate', 'mean'),
        avg_trades     = ('trade_count', 'mean'),
        avg_size_usd   = ('avg_size_usd', 'mean')
    ).round(2)
    st.dataframe(summary, use_container_width=True)

# ── PAGE 4 — Trader Segments ───────────────────
elif page == "Trader Segments":
    st.title("👥 Trader Segments Analysis")

    segment = st.selectbox("Select Segment to Explore", [
        "Risk Segment (High vs Low)",
        "Frequency Segment (Frequent vs Infrequent)",
        "Performance Segment (Winners vs Inconsistent)"
    ])

    if segment == "Risk Segment (High vs Low)":
        col = 'risk_segment'
        title = 'Risk Segment'
        colors = ['#e74c3c', '#2ecc71']

    elif segment == "Frequency Segment (Frequent vs Infrequent)":
        col = 'frequency_segment'
        title = 'Frequency Segment'
        colors = ['#3498db', '#e67e22']

    else:
        col = 'performance_segment'
        title = 'Performance Segment'
        colors = ['#9b59b6', '#1abc9c']

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Net PnL by {title} and Sentiment")
        fig, ax = plt.subplots(figsize=(6, 4))
        seg_data = df_daily.groupby(
            [col, 'sentiment_binary'])['net_pnl'].mean().unstack()
        seg_data.T.plot(kind='bar', ax=ax, color=colors)
        ax.set_title(f'Net PnL by {title}')
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Average Net PnL (USD)')
        plt.xticks(rotation=0)
        plt.legend(title=title)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader(f"Summary Statistics")
        seg_summary = df_daily.groupby(col).agg(
            avg_net_pnl  = ('net_pnl', 'mean'),
            avg_win_rate = ('win_rate', 'mean'),
            avg_trades   = ('trade_count', 'mean'),
            avg_size_usd = ('avg_size_usd', 'mean')
        ).round(2)
        st.dataframe(seg_summary, use_container_width=True)

# ── PAGE 5 — Strategy Recommendations ─────────
elif page == "Strategy Recommendations":
    st.title("🎯 Strategy Recommendations")
    st.markdown("Based on data analysis of 104,272 trades across Fear, Neutral and Greed market conditions.")

    st.markdown("---")
    st.subheader("Strategy 1 — Fear Day Aggressive Trading")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        During **Fear days**, High Risk + Frequent traders should:
        - Maximise position sizes (data shows 2x bigger = 2.7x more profit)
        - Increase trade frequency (70 trades/day vs 54 on Greed days)
        - Maintain short bias (prices falling = short sellers profit)
        """)
    with col2:
        st.metric("High Risk Fear PnL", "$11,538")
        st.metric("Low Risk Fear PnL", "$4,266")
        st.metric("Difference", "2.7x")

    st.markdown("---")
    st.subheader("Strategy 2 — Greed Day Position Holding")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        During **Greed days**, Inconsistent/aggressive traders should:
        - Widen profit targets and hold positions longer
        - Allow winning trades to run further
        - Accept higher volatility for higher returns
        """)
    with col2:
        st.metric("Inconsistent Greed PnL", "$8,357")
        st.metric("Consistent Greed PnL", "$3,534")
        st.metric("Difference", "2.4x")

    st.markdown("---")
    st.subheader("Strategy 3 — Avoid Neutral Days")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        During **Neutral days**, all traders should:
        - Reduce trading activity significantly
        - Low Risk traders especially should sit out
        - Transaction fees eat into minimal profits
        """)
    with col2:
        st.metric("Low Risk Neutral PnL", "$1,357")
        st.metric("Low Risk Fear PnL", "$4,266")
        st.metric("Difference", "3x worse")

    st.markdown("---")
    st.subheader("Model Findings")
    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        **Random Forest Model**
        - Accuracy: 96%
        - Correctly identified 81% of losing days
        - Only 12 mistakes out of 338 predictions
        """)
    with col2:
        st.success("""
        **Key Predictor**
        - Win rate = 72% importance
        - Trader skill > Market conditions
        - Sentiment value = only 6.6% importance
        """)
