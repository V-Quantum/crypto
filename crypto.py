import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
from datetime import datetime, timedelta
import plotly.express as px
from pycoingecko import CoinGeckoAPI

# Inisialisasi CoinGecko API
cg = CoinGeckoAPI()

@st.cache_data(ttl=3600)  # Cache data selama 1 jam
def get_all_coins():
    """Mengambil daftar semua cryptocurrency dari CoinGecko"""
    try:
        coins = cg.get_coins_markets(vs_currency='usd', order='market_cap_desc', per_page=250, page=1)
        return pd.DataFrame(coins)[['id', 'symbol', 'name', 'market_cap_rank', 'current_price', 'market_cap']]
    except Exception as e:
        st.error(f"Error fetching coin list: {str(e)}")
        return None

def get_crypto_data(crypto_id, days=365):
    """Mengambil data historis cryptocurrency dari CoinGecko"""
    try:
        data = cg.get_coin_market_chart_by_id(id=crypto_id, vs_currency='usd', days=days)
        prices = data['prices']
        df = pd.DataFrame(prices, columns=['Date', 'Price'])
        df['Date'] = pd.to_datetime(df['Date'], unit='ms')
        return df
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def prepare_data_for_prophet(df):
    """Menyiapkan data untuk model Prophet"""
    prophet_df = df.rename(columns={'Date': 'ds', 'Price': 'y'})
    return prophet_df

def train_prophet_model(df, periods=30):
    """Melatih model Prophet dan membuat prediksi"""
    model = Prophet(
        daily_seasonality=True,
        yearly_seasonality=True,
        weekly_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10
    )
    model.fit(df)
    
    future_dates = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future_dates)
    return forecast

def plot_forecast(original_df, forecast_df, crypto_name):
    """Membuat visualisasi data asli dan prediksi"""
    fig = go.Figure()

    # Plot data asli
    fig.add_trace(go.Scatter(
        x=original_df['Date'],
        y=original_df['Price'],
        name='Harga Aktual',
        line=dict(color='blue')
    ))

    # Plot prediksi
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat'],
        name='Prediksi',
        line=dict(color='red')
    ))

    # Plot interval kepercayaan
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat_upper'],
        fill=None,
        mode='lines',
        line_color='rgba(255,0,0,0.2)',
        name='Upper Bound'
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat_lower'],
        fill='tonexty',
        mode='lines',
        line_color='rgba(255,0,0,0.2)',
        name='Lower Bound'
    ))

    fig.update_layout(
        title=f'Prediksi Harga {crypto_name}',
        xaxis_title='Tanggal',
        yaxis_title='Harga (USD)',
        hovermode='x unified',
        template='plotly_white'
    )

    return fig

def format_number(number):
    """Format angka dengan pemisah ribuan dan 2 desimal"""
    return f"${number:,.2f}"

def main():
    st.set_page_config(page_title="Global Crypto Price Forecasting", layout="wide")
    
    # Header
    st.title("🌍 Global Crypto Price Forecasting")
    st.markdown("""
    Aplikasi ini memprediksi harga cryptocurrency menggunakan model Prophet.
    Data diambil dari CoinGecko API. Anda dapat mencari dan memilih cryptocurrency apapun yang terdaftar di CoinGecko.
    """)
    
    # Sidebar
    st.sidebar.header("Parameter Prediksi")
    
    # Ambil daftar semua cryptocurrency
    with st.spinner("Mengambil daftar cryptocurrency..."):
        all_coins_df = get_all_coins()
    
    if all_coins_df is not None:
        # Search box untuk cryptocurrency
        search_term = st.sidebar.text_input("Cari Cryptocurrency (nama atau simbol)")
        
        # Filter cryptocurrency berdasarkan pencarian
        if search_term:
            filtered_coins = all_coins_df[
                all_coins_df['name'].str.contains(search_term, case=False) |
                all_coins_df['symbol'].str.contains(search_term, case=False)
            ]
        else:
            filtered_coins = all_coins_df
        
        # Tampilkan hasil pencarian dalam format yang informatif
        coin_options = [
            f"{row['name']} ({row['symbol'].upper()}) - Rank #{row['market_cap_rank']}" 
            for _, row in filtered_coins.iterrows()
        ]
        
        if len(coin_options) > 0:
            selected_coin = st.sidebar.selectbox(
                "Pilih Cryptocurrency",
                coin_options
            )
            
            # Dapatkan ID cryptocurrency yang dipilih
            selected_idx = coin_options.index(selected_coin)
            selected_coin_id = filtered_coins.iloc[selected_idx]['id']
            selected_coin_name = filtered_coins.iloc[selected_idx]['name']
            
            # Parameter tambahan
            days_historical = st.sidebar.slider(
                "Jumlah Hari Data Historis",
                min_value=30,
                max_value=365,
                value=180
            )
            
            forecast_days = st.sidebar.slider(
                "Jumlah Hari Prediksi",
                min_value=7,
                max_value=90,
                value=30
            )
            
            # Tampilkan informasi coin yang dipilih
            col1, col2, col3 = st.columns(3)
            col1.metric(
                "Harga Saat Ini",
                format_number(filtered_coins.iloc[selected_idx]['current_price'])
            )
            col2.metric(
                "Market Cap Rank",
                f"#{filtered_coins.iloc[selected_idx]['market_cap_rank']}"
            )
            col3.metric(
                "Market Cap",
                format_number(filtered_coins.iloc[selected_idx]['market_cap'])
            )
            
            # Main content
            if st.button("Mulai Prediksi"):
                with st.spinner("Mengambil data historis..."):
                    df = get_crypto_data(selected_coin_id, days_historical)
                    
                    if df is not None:
                        # Tampilkan info perubahan harga
                        price_change = ((df['Price'].iloc[-1] - df['Price'].iloc[0]) / df['Price'].iloc[0]) * 100
                        st.metric(
                            f"Perubahan Harga ({days_historical} hari)",
                            f"{price_change:.2f}%"
                        )
                        
                        # Siapkan data untuk Prophet
                        prophet_df = prepare_data_for_prophet(df)
                        
                        with st.spinner("Melatih model dan membuat prediksi..."):
                            # Latih model dan buat prediksi
                            forecast = train_prophet_model(prophet_df, periods=forecast_days)
                        
                        # Plot hasil
                        st.subheader("Grafik Prediksi")
                        fig = plot_forecast(df, forecast, selected_coin_name)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Tampilkan prediksi terakhir
                        st.subheader("Prediksi Harga")
                        last_prediction = forecast.iloc[-1]
                        pred_cols = st.columns(3)
                        
                        pred_cols[0].metric(
                            "Prediksi",
                            format_number(last_prediction['yhat'])
                        )
                        pred_cols[1].metric(
                            "Lower Bound",
                            format_number(last_prediction['yhat_lower'])
                        )
                        pred_cols[2].metric(
                            "Upper Bound",
                            format_number(last_prediction['yhat_upper'])
                        )
                        
                        # Tampilkan tabel prediksi
                        st.subheader("Tabel Prediksi")
                        forecast_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days)
                        forecast_table.columns = ['Tanggal', 'Prediksi', 'Batas Bawah', 'Batas Atas']
                        forecast_table['Prediksi'] = forecast_table['Prediksi'].apply(format_number)
                        forecast_table['Batas Bawah'] = forecast_table['Batas Bawah'].apply(format_number)
                        forecast_table['Batas Atas'] = forecast_table['Batas Atas'].apply(format_number)
                        st.dataframe(forecast_table)
        else:
            st.sidebar.warning("Tidak ada cryptocurrency yang sesuai dengan pencarian")

if __name__ == "__main__":
    main()