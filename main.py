import time
import sqlite3
from datetime import datetime, date, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

from data.loader import (
    get_stock_data, 
    get_company_info, 
    get_asset_currency,
    get_exchange_rate_series, 
    get_central_bank_rate, 
    get_cumulative_split_factor,
    search_assets
)

from data.database import (
    create_user, check_login, save_portfolio_db, load_portfolios_db,
    delete_portfolio_db, delete_user_full, approve_user
)

from analysis import transformations, risk, portfolio, technical

from utils.helpers import (
    to_scalar, 
    get_price_at_date, 
    asset_selector, 
    render_news_section,
    render_options_analysis
)

from utils.strategy_manager import render_strategy_builder
from utils.strategy_manager import load_strategies
from utils.backtester import run_backtest_engine

st.set_page_config(page_title="Vektor", layout="wide")

def run_dashboard(current_user):
    st.sidebar.title(f"User: {current_user}")
    
    if current_user == "Admin": 
        st.sidebar.markdown("---")
        st.sidebar.subheader("Admin Panel")
        conn = sqlite3.connect("Vektor.db")
        
        pending = pd.read_sql_query("SELECT username FROM users WHERE is_approved = 0", conn)
        if not pending.empty:
            st.sidebar.error(f"{len(pending)} Anfragen!")
            for idx, row in pending.iterrows():
                u_new = row['username']
                c1, c2 = st.sidebar.columns([3, 1])
                c1.write(f"**{u_new}**")
                if c2.button("OK", key=f"app_{u_new}"):
                    approve_user(u_new)
                    st.sidebar.success("Freigeschaltet!")
                    time.sleep(1)
                    st.rerun()
        else:
            st.sidebar.success("Keine Anfragen.")
        
        if st.sidebar.checkbox("User löschen?"):
            all_users = pd.read_sql_query("SELECT username FROM users", conn)['username'].tolist()
            del_user = st.sidebar.selectbox("Wen löschen?", ["Wählen..."] + all_users)
            if del_user != "Wählen..." and del_user != current_user:
                if st.sidebar.button(f"{del_user} LÖSCHEN"):
                    delete_user_full(del_user)
                    st.rerun()
        conn.close()

    if st.sidebar.button("Ausloggen"):
        st.session_state.logged_in_user = None
        st.rerun()
        
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("Einstellungen")
    pf_currency = st.sidebar.selectbox("Währung", ["EUR", "USD", "CHF", "GBP"], index=0)
    
    data_res = st.sidebar.select_slider(
        "Daten-Basis", 
        options=["Täglich (Langzeit)", "Intraday (15 Min)", "Live (1 Min)"], 
        value="Täglich (Langzeit)"
    )
    
    if data_res == "Live (1 Min)":
        load_period = "5d"
        load_interval = "1m"
    elif data_res == "Intraday (15 Min)":
        load_period = "60d"
        load_interval = "15m"
    else:
        load_period = "max"
        load_interval = "1d"

    show_only_today = False
    if load_interval != "1d":
        show_only_today = st.sidebar.checkbox("Nur heutigen Tag anzeigen", value=False)

    st.sidebar.markdown("---")

    chart_mode = st.sidebar.radio("Chart Anzeige", ["Absoluter Kurs/Wert", "Indexiert (Start=100)"], index=0)

    if not show_only_today:
        st.sidebar.markdown("**Zeitraum**")
        time_range_mode = st.sidebar.radio("Range", ["Maximal / Portfolio-Start", "Benutzerdefiniert"], label_visibility="collapsed")
        
        custom_start = None
        custom_end = None
        
        if time_range_mode == "Benutzerdefiniert":
            c_d1, c_d2 = st.sidebar.columns(2)
            default_start = date.today() - timedelta(days=365)
            custom_start = c_d1.date_input("Von", value=default_start)
            custom_end = c_d2.date_input("Bis", value=date.today())
    else:
        st.sidebar.info("Modus: Nur Heute")
        time_range_mode = "Maximal / Portfolio-Start"
        custom_start = None
        custom_end = None

    st.sidebar.markdown("---")

    auto_risk_free = get_central_bank_rate(pf_currency)
    risk_free = st.sidebar.number_input(
        f"Risk-Free Rate ({pf_currency})", 
        min_value=0.0, 
        max_value=0.2, 
        value=float(auto_risk_free), 
        step=0.0025,
        format="%.4f"
    )
    
    show_advanced_stats = st.sidebar.checkbox("Erweiterte Statistiken", value=False)
    show_options = st.sidebar.checkbox("Optionen anzeigen", value=False)
    
    st.sidebar.markdown("---")

    mode = st.sidebar.radio("Modus wählen", ["Einzel-Analyse", "Vergleich", "Portfolio Manager"]) #strategy Lab raus, weil discontinued
    st.sidebar.markdown("---")

    benchmarks = {"US S&P 500": "^GSPC", "MSCI World": "URTH", "Bitcoin": "BTC-USD", "Gold": "GC=F"}
    
    main_series = None
    main_prices = None 
    main_name = ""
    
    compare_series = None
    compare_prices = None
    compare_name = ""
    
    main_df = None

    news_sources = {}

    default_data_start = None

    if mode == "Portfolio Manager":
        st.title(f"Portfolio Manager ({pf_currency})")
        saved_portfolios = load_portfolios_db(current_user)
        pf_names = ["Neues Portfolio erstellen..."] + list(saved_portfolios.keys())
        
        with st.container(border=True):
            st.subheader("Portfolio Auswahl")
            c_pf1, c_pf2 = st.columns([3, 1])
            with c_pf1:
                selected_pf_name = st.selectbox("Portfolio wählen", pf_names, label_visibility="collapsed")
            with c_pf2:
                if selected_pf_name != "Neues Portfolio erstellen...":
                    if st.button("Portfolio Löschen", use_container_width=True):
                        delete_portfolio_db(current_user, selected_pf_name)
                        st.session_state.current_pf_data = {}
                        st.rerun()
        
        if 'current_pf_data' not in st.session_state or st.session_state.get('last_loaded_pf') != selected_pf_name:
            if selected_pf_name != "Neues Portfolio erstellen...":
                st.session_state.current_pf_data = saved_portfolios[selected_pf_name]
            else:
                st.session_state.current_pf_data = {} 
            st.session_state.last_loaded_pf = selected_pf_name

        st.write("")
        with st.container(border=True):
            st.subheader("Transaktion erfassen")
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                search_q = st.text_input("Asset suchen", key="pf_add_search")
                found_ticker = None
                if search_q:
                    res = search_assets(search_q)
                    if res:
                        sel = st.selectbox("Ergebnis", res, label_visibility="collapsed")
                        if "(" in sel and ")" in sel:
                            found_ticker = sel.split('(')[1].split(')')[0]
                        else:
                            found_ticker = sel
            
            transaction_type = "Kauf"
            if found_ticker and found_ticker in st.session_state.current_pf_data:
                transaction_type = st.radio("Aktion:", ["Kauf", "Verkauf"], horizontal=True)

            with c2:
                input_type = st.radio("Eingabe als:", [f"Geldwert ({pf_currency})", "Stückzahl"], horizontal=True)
            with c3:
                date_val = st.date_input("Datum", value=datetime.today())

            val_col, btn_col = st.columns([2, 1])
            with val_col:
                amount_val = st.number_input(f"Menge", min_value=0.01, value=1000.0)
            
            with btn_col:
                st.write(""); st.write("")
                btn_label = "Buchen" if "Kauf" in transaction_type else "Verkaufen"
                if st.button(btn_label, use_container_width=True):
                    if found_ticker:
                        with st.spinner("Lade Preis & Währung..."):
                            df_temp = get_stock_data(found_ticker, period="max", interval="1d")
                            asset_curr = get_asset_currency(found_ticker)
                            
                        if df_temp is not None:
                            raw_price = get_price_at_date(df_temp['Close'], str(date_val))
                            
                            if raw_price:
                                conversion_rate = 1.0
                                if asset_curr != pf_currency:
                                    fx_hist = get_exchange_rate_series(asset_curr, pf_currency, period="max")
                                    if fx_hist is not None:
                                        fx_at_date = get_price_at_date(fx_hist, str(date_val))
                                        if fx_at_date:
                                            conversion_rate = fx_at_date
                                
                                price_in_pf_curr = raw_price * conversion_rate
                                
                                if input_type == "Stückzahl":
                                    shares_t = amount_val
                                    invest_t = shares_t * price_in_pf_curr
                                else:
                                    shares_t = amount_val / price_in_pf_curr
                                    invest_t = amount_val
                                
                                curr = st.session_state.current_pf_data.get(found_ticker, {"shares": 0.0, "invested": 0.0, "buy_date": str(date_val)})
                                
                                history_dates = curr.get("buy_history", [curr.get("buy_date", str(date_val))])
                                if not isinstance(history_dates, list): history_dates = [str(history_dates)]

                                if "Kauf" in transaction_type:
                                    new_s = curr["shares"] + shares_t
                                    new_i = curr["invested"] + invest_t
                                    new_avg = new_i / new_s if new_s > 0 else 0
                                    
                                    if str(date_val) not in history_dates:
                                        history_dates.append(str(date_val))
                                        history_dates.sort()

                                    st.session_state.current_pf_data[found_ticker] = {
                                        "shares": new_s, 
                                        "invested": new_i, 
                                        "buy_date": curr["buy_date"], 
                                        "buy_history": history_dates,
                                        "price_at_buy": new_avg
                                    }
                                    st.success("Gebucht!")
                                else:
                                    if shares_t > curr["shares"]:
                                        st.error("Nicht genug Bestand.")
                                    else:
                                        ratio = shares_t / curr["shares"]
                                        cost_sold = curr["invested"] * ratio
                                        new_s = curr["shares"] - shares_t
                                        new_i = curr["invested"] - cost_sold
                                        if new_s < 0.0001:
                                            del st.session_state.current_pf_data[found_ticker]
                                        else:
                                            st.session_state.current_pf_data[found_ticker] = {
                                                "shares": new_s, 
                                                "invested": new_i, 
                                                "buy_date": curr["buy_date"], 
                                                "buy_history": history_dates,
                                                "price_at_buy": curr.get("price_at_buy", 0)
                                            }
                                        st.success("Verkauft!")
                                st.rerun()
                            else:
                                st.error("Kein Preis zum Datum.")
                        else:
                            st.error("Fehler Daten.")
                    else:
                        st.warning("Erst suchen.")

        if st.session_state.current_pf_data:
            st.markdown("---")
            table_rows = []
            tickers = list(st.session_state.current_pf_data.keys())
            df_all = pd.DataFrame()
            shares_dict_for_chart = {}
            total_inv = 0.0
            total_val = 0.0
            all_buy_dates = []
            
            holdings_for_news = []

            for t in tickers:
                d = st.session_state.current_pf_data[t]
                df_h = get_stock_data(t, period=load_period, interval=load_interval)
                
                if df_h is not None:
                    if df_h.index.tz is not None:
                        df_h.index = df_h.index.tz_localize(None)

                    asset_curr = get_asset_currency(t)
                    price_series = df_h['Close']
                    
                    if asset_curr != pf_currency:
                        fx_series = get_exchange_rate_series(asset_curr, pf_currency)
                        if load_interval != "1d":
                            fx_series = fx_series.reindex(price_series.index, method='ffill')
                        price_series = portfolio.convert_to_portfolio_currency(price_series, fx_series)
                        
                    curr_p = to_scalar(price_series.iloc[-1])
                    
                    buy_date = d.get('buy_date', datetime.today().strftime('%Y-%m-%d'))
                    all_buy_dates.append(buy_date)
                    
                    split_factor = get_cumulative_split_factor(t, buy_date)
                    real_shares = d['shares'] * split_factor
                    shares_dict_for_chart[t] = real_shares
                    
                    val_n = real_shares * curr_p
                    holdings_for_news.append((t, val_n))

                    perf_rel = (val_n / d['invested']) - 1 if d['invested'] > 0 else 0
                    total_inv += d['invested']
                    total_val += val_n
                    
                    shares_display = f"{real_shares:.2f}"
                    if split_factor > 1.01:
                        shares_display += f" (x{split_factor:g})"

                    raw_history = d.get('buy_history', [d.get('buy_date', '-')])
                    if isinstance(raw_history, list):
                        buy_dates_display = ", ".join(raw_history)
                    else:
                        buy_dates_display = str(raw_history)

                    table_rows.append({
                        "Ticker": t, 
                        "Kaufdaten": buy_dates_display,
                        "Orig. Währung": asset_curr,
                        "Stück": shares_display, 
                        "Invest": f"{d['invested']:.2f}", 
                        "Wert": f"{val_n:.2f}", 
                        "G/V %": f"{perf_rel:.2%}"
                    })
                    df_all[t] = price_series
            
            total_gain = total_val - total_inv
            tot_rel = (total_val / total_inv) - 1 if total_inv > 0 else 0
            
            c1, c2, c3 = st.columns(3)
            c1.metric(f"Portfoliowert ({pf_currency})", f"{total_val:,.2f}")
            c2.metric("Gewinn", f"{total_gain:,.2f}", delta=f"{tot_rel:.2%}")
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
            
            col_s1, col_s2, col_s3 = st.columns([2, 1, 1])
            
            with col_s1:
                save_name = st.text_input("Name:", value=selected_pf_name if "Neues" not in selected_pf_name else "")
            
            with col_s2: 
                st.write(""); st.write("")
                if st.button("Speichern", use_container_width=True):
                    if save_name:
                        save_portfolio_db(current_user, save_name, st.session_state.current_pf_data)
                        st.success("Gespeichert!")
                        st.rerun()
            
            with col_s3:
                st.write(""); st.write("")
                if "Neues" not in selected_pf_name:
                    if st.button("Löschen", use_container_width=True):
                        delete_portfolio_db(current_user, selected_pf_name)
                        st.session_state.current_pf_data = {}
                        st.rerun()
            
            df_all = df_all.ffill() 
            df_all = df_all.fillna(0.0)
            
            if load_interval == "1d" and all_buy_dates and not show_only_today:
                min_buy_date = min(all_buy_dates) 
                default_data_start = pd.to_datetime(min_buy_date)
                df_all = df_all[df_all.index >= default_data_start]

            df_all = df_all.loc[(df_all.sum(axis=1) > 0)]
            
            if not df_all.empty:
                pf_val, pf_ret = portfolio.calculate_buy_and_hold_series(df_all, shares_dict_for_chart)
                
                main_prices = pf_val 
                main_series = pf_ret
                main_name = f"Portfolio '{save_name}'"
                
                bench = asset_selector("Benchmark", "bench_pf", predefined_options=benchmarks)
                
                if bench:
                    bd_code = benchmarks.get(bench, bench) if bench in benchmarks else bench
                    news_sources[bd_code] = "Markt & Macro"
                
                holdings_for_news.sort(key=lambda x: x[1], reverse=True)
                
                for t, v in holdings_for_news:
                    news_sources[t] = f"{t}"

                if bench:
                    bd = get_stock_data(bench, period=load_period, interval=load_interval)
                    if bd is not None:
                        if bd.index.tz is not None:
                            bd.index = bd.index.tz_localize(None)

                        b_curr = get_asset_currency(bench)
                        b_prices = bd['Close']

                        if b_curr != pf_currency:
                            fx_b = get_exchange_rate_series(b_curr, pf_currency)
                            if load_interval != "1d":
                                fx_b = fx_b.reindex(b_prices.index, method='ffill')
                            b_prices = portfolio.convert_to_portfolio_currency(b_prices, fx_b)
                        
                        compare_prices = b_prices
                        compare_series = transformations.calculate_log_returns(b_prices)
                        compare_name = bench
        else:
            st.info("Leeres Portfolio.")

    elif mode == "Vergleich":
        st.title(f"Vergleich ({pf_currency})")
        with st.container(border=True):
            c1, c2 = st.columns(2)
            with c1:
                t1 = asset_selector("Asset 1", "a1", default_symbol="AAPL")
            with c2:
                t2 = asset_selector("Asset 2", "a2", default_symbol="MSFT")
        
        if t1 and t2:
            news_sources[t1] = "Asset 1"
            news_sources[t2] = "Asset 2"

            d1 = get_stock_data(t1, period=load_period, interval=load_interval)
            d2 = get_stock_data(t2, period=load_period, interval=load_interval)
            
            if d1 is not None and d2 is not None:
                if d1.index.tz is not None: d1.index = d1.index.tz_localize(None)
                if d2.index.tz is not None: d2.index = d2.index.tz_localize(None)

                c1 = get_asset_currency(t1)
                p1 = d1['Close']
                if c1 != pf_currency:
                    fx1 = get_exchange_rate_series(c1, pf_currency)
                    if load_interval != "1d": fx1 = fx1.reindex(p1.index, method='ffill')
                    p1 = portfolio.convert_to_portfolio_currency(p1, fx1)
                
                c2 = get_asset_currency(t2)
                p2 = d2['Close']
                if c2 != pf_currency:
                    fx2 = get_exchange_rate_series(c2, pf_currency)
                    if load_interval != "1d": fx2 = fx2.reindex(p2.index, method='ffill')
                    p2 = portfolio.convert_to_portfolio_currency(p2, fx2)

                main_prices = p1
                main_series = transformations.calculate_log_returns(p1)
                main_series = main_series.replace([np.inf, -np.inf], np.nan).dropna()
                main_name = t1
                
                compare_prices = p2
                compare_series = transformations.calculate_log_returns(p2)
                compare_name = t2

    else:
        st.title(f"Einzel-Analyse ({pf_currency})")
        with st.container(border=True):
            t1 = asset_selector("Asset", "single", default_symbol="AAPL")
        
        if t1:
            news_sources[t1] = "Asset"

            info = get_company_info(t1)
            
            d1 = get_stock_data(t1, period=load_period, interval=load_interval)
            if d1 is not None:
                if d1.index.tz is not None:
                    d1.index = d1.index.tz_localize(None)

                c1 = get_asset_currency(t1)
                p1 = d1['Close']
                main_df = d1.copy()
                
                if c1 != pf_currency:
                    fx1 = get_exchange_rate_series(c1, pf_currency)
                    if load_interval != "1d": 
                        fx1 = fx1.reindex(p1.index, method='ffill')
                    
                    p1 = portfolio.convert_to_portfolio_currency(p1, fx1)
                    for col in ['Open', 'High', 'Low', 'Close']:
                        if col in main_df.columns:
                            main_df[col] = portfolio.convert_to_portfolio_currency(main_df[col], fx1)
                
                cur_val = to_scalar(p1.iloc[-1])
                head_c1, head_c2 = st.columns([3, 1])
                with head_c1:
                    st.subheader(info.get('longName', t1))
                    st.caption(f"{t1} | {info.get('sector','-')}")
                with head_c2:
                    st.metric(f"Kurs ({pf_currency})", f"{cur_val:,.2f}")

                if not p1.empty:
                    last_ts = p1.index[-1]
                    st.caption(f"Letzter Datenpunkt: {last_ts.strftime('%d.%m.%Y %H:%M')}")

                valid_indices = p1.index[p1 > 0]
                if not valid_indices.empty:
                    first_valid = valid_indices[0]
                    p1 = p1.loc[first_valid:]
                    main_df = main_df.loc[first_valid:]
                
                main_prices = p1
                main_series = transformations.calculate_log_returns(p1)
                main_series = main_series.replace([np.inf, -np.inf], np.nan).dropna()
                main_name = t1

    if main_series is not None:
        
        display_start = None
        display_end = None
        
        if show_only_today:
            display_start = pd.Timestamp.now().normalize()
        elif time_range_mode == "Benutzerdefiniert":
            if custom_start and custom_end:
                display_start = pd.to_datetime(custom_start)
                display_end = pd.to_datetime(custom_end)
        else:
            if default_data_start and load_interval == "1d":
                display_start = default_data_start
        
        y_data_filtered = main_prices
        if display_start:
            y_data_filtered = y_data_filtered[y_data_filtered.index >= display_start]
        if display_end:
            y_data_filtered = y_data_filtered[y_data_filtered.index <= display_end]
            
        fig = go.Figure()

        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            chart_type_sel = st.selectbox("Typ", ["Line", "Candle", "OHLC"], key="ctype")
        with col_c2:
            is_log = st.checkbox("Log-Skala", value=False)
        with col_c3:
            indicators_sel = st.multiselect("Indikatoren", ["SMA 1", "SMA 2", "Bollinger", "RSI"], key="inds")

        sma1_val, sma2_val = 50, 200
        if "SMA 1" in indicators_sel or "SMA 2" in indicators_sel:
            c_s1, c_s2, c_s3 = st.columns(3)
            if "SMA 1" in indicators_sel:
                sma1_val = c_s1.number_input("Zeitraum SMA 1", min_value=2, max_value=500, value=50)
            if "SMA 2" in indicators_sel:
                sma2_val = c_s2.number_input("Zeitraum SMA 2", min_value=2, max_value=500, value=200)

        is_index_mode = chart_mode == "Indexiert (Start=100)"
        is_pro_mode = (mode == "Einzel-Analyse" and main_df is not None and not is_index_mode)

        has_rsi = "RSI" in indicators_sel
        
        if has_rsi and is_pro_mode:
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08, 
                                row_heights=[0.6, 0.2, 0.2],
                                subplot_titles=(f"Kurs ({pf_currency})", "Volumen", "RSI (14)"))
        elif is_pro_mode:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, 
                                row_heights=[0.7, 0.3],
                                subplot_titles=(f"Kurs ({pf_currency})", "Volumen"))
        elif has_rsi:
             fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, 
                                row_heights=[0.8, 0.2],
                                subplot_titles=(f"Kurs/Index", "RSI (14)"))
        else:
             fig = make_subplots(rows=1, cols=1)

        if is_pro_mode:
            df_plot = main_df
            if display_start: df_plot = df_plot[df_plot.index >= display_start]
            if display_end: df_plot = df_plot[df_plot.index <= display_end]
            
            if df_plot.empty:
                st.warning("Keine Daten im gewählten Zeitraum. (Möglicherweise hat der Markt heute noch nicht geöffnet.)")
            
            if chart_type_sel == "Candle":
                fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], name=main_name), row=1, col=1)
            elif chart_type_sel == "OHLC":
                fig.add_trace(go.Ohlc(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], name=main_name), row=1, col=1)
            else:
                fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Close'], mode='lines', name=main_name), row=1, col=1)
            
            if 'Volume' in df_plot.columns:
                fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['Volume'], name="Volumen", marker_color='#00FF00', marker_line_width=0), row=2, col=1)
            
            close_series = df_plot['Close']
            
            if "SMA 1" in indicators_sel:
                sma1 = technical.calculate_sma(close_series, sma1_val)
                fig.add_trace(go.Scatter(x=df_plot.index, y=sma1, mode='lines', name=f"SMA {sma1_val}", line=dict(color='orange', width=1)), row=1, col=1)
                
            if "SMA 2" in indicators_sel:
                sma2 = technical.calculate_sma(close_series, sma2_val)
                fig.add_trace(go.Scatter(x=df_plot.index, y=sma2, mode='lines', name=f"SMA {sma2_val}", line=dict(color='blue', width=1)), row=1, col=1)
                
            if "Bollinger" in indicators_sel:
                upper, lower = technical.calculate_bollinger_bands(close_series)
                fig.add_trace(go.Scatter(x=df_plot.index, y=upper, mode='lines', name="BB Upper", line=dict(width=0), showlegend=False), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_plot.index, y=lower, mode='lines', name="Bollinger", fill='tonexty', fillcolor='rgba(0, 100, 255, 0.1)', line=dict(width=0)), row=1, col=1)

            if has_rsi:
                rsi_vals = technical.calculate_rsi(close_series)
                rsi_row = 3
                fig.add_trace(go.Scatter(x=df_plot.index, y=rsi_vals, mode='lines', name="RSI", line=dict(color='purple')), row=rsi_row, col=1)
                fig.add_hline(y=70, line_dash="dot", line_color="red", row=rsi_row, col=1)
                fig.add_hline(y=30, line_dash="dot", line_color="green", row=rsi_row, col=1)

            if compare_series is not None and compare_prices is not None:
                comp_filtered = compare_prices
                if display_start: comp_filtered = comp_filtered[comp_filtered.index >= display_start]
                if display_end: comp_filtered = comp_filtered[comp_filtered.index <= display_end]
                
                common_idx = df_plot.index.intersection(comp_filtered.index)
                if len(common_idx) > 0:
                    comp_aligned = comp_filtered.loc[common_idx]
                    fig.add_trace(go.Scatter(x=common_idx, y=comp_aligned, name=compare_name, line=dict(dash='dot', color='yellow')), row=1, col=1)
                    
                    if "SMA 1" in indicators_sel:
                        sma1_c = technical.calculate_sma(comp_aligned, sma1_val)
                        fig.add_trace(go.Scatter(x=common_idx, y=sma1_c, mode='lines', name=f"SMA {sma1_val} ({compare_name})", line=dict(color='orange', width=1, dash='dot')), row=1, col=1)
                    if "SMA 2" in indicators_sel:
                        sma2_c = technical.calculate_sma(comp_aligned, sma2_val)
                        fig.add_trace(go.Scatter(x=common_idx, y=sma2_c, mode='lines', name=f"SMA {sma2_val} ({compare_name})", line=dict(color='blue', width=1, dash='dot')), row=1, col=1)

        else:
            if is_index_mode:
                if len(y_data_filtered) > 0:
                    first_val = to_scalar(y_data_filtered.iloc[0])
                    y_data_plot = 100 * (y_data_filtered / first_val) if first_val > 0 else y_data_filtered
                else:
                    y_data_plot = y_data_filtered
                t_y = "Index (100)"
            else:
                y_data_plot = y_data_filtered
                t_y = f"Kurs/Wert ({pf_currency})"
            
            fig.add_trace(go.Scatter(x=y_data_plot.index, y=y_data_plot, mode='lines', name=main_name), row=1, col=1)
            
            if "SMA 1" in indicators_sel:
                sma1 = technical.calculate_sma(y_data_plot, sma1_val)
                fig.add_trace(go.Scatter(x=y_data_plot.index, y=sma1, mode='lines', name=f"SMA {sma1_val}", line=dict(color='orange', width=1)), row=1, col=1)
            
            if "SMA 2" in indicators_sel:
                sma2 = technical.calculate_sma(y_data_plot, sma2_val)
                fig.add_trace(go.Scatter(x=y_data_plot.index, y=sma2, mode='lines', name=f"SMA {sma2_val}", line=dict(color='blue', width=1)), row=1, col=1)
            
            if "Bollinger" in indicators_sel:
                upper, lower = technical.calculate_bollinger_bands(y_data_plot)
                fig.add_trace(go.Scatter(x=y_data_plot.index, y=upper, mode='lines', name="BB Upper", line=dict(width=0), showlegend=False), row=1, col=1)
                fig.add_trace(go.Scatter(x=y_data_plot.index, y=lower, mode='lines', name="Bollinger", fill='tonexty', fillcolor='rgba(0, 100, 255, 0.1)', line=dict(width=0)), row=1, col=1)

            if has_rsi:
                rsi_vals = technical.calculate_rsi(y_data_plot)
                fig.add_trace(go.Scatter(x=y_data_plot.index, y=rsi_vals, mode='lines', name="RSI", line=dict(color='purple')), row=2, col=1)
                fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)

            if compare_series is not None and compare_prices is not None:
                comp_filtered = compare_prices
                if display_start: comp_filtered = comp_filtered[comp_filtered.index >= display_start]
                if display_end: comp_filtered = comp_filtered[comp_filtered.index <= display_end]
                
                common_idx = y_data_filtered.index.intersection(comp_filtered.index)
                if len(common_idx) > 0:
                    y_aligned = y_data_filtered.loc[common_idx]
                    comp_aligned = comp_filtered.loc[common_idx]
                    
                    if is_index_mode:
                        c_start = to_scalar(comp_aligned.iloc[0])
                        comp_data_plot = 100 * (comp_aligned / c_start) if c_start > 0 else comp_aligned
                    else:
                        comp_data_plot = comp_aligned

                    fig.add_trace(go.Scatter(x=common_idx, y=comp_data_plot, name=compare_name, line=dict(dash='dot')), row=1, col=1)
                    
                    if "SMA 1" in indicators_sel:
                        sma1_c = technical.calculate_sma(comp_data_plot, sma1_val)
                        fig.add_trace(go.Scatter(x=common_idx, y=sma1_c, mode='lines', name=f"SMA {sma1_val} ({compare_name})", line=dict(color='orange', width=1, dash='dot')), row=1, col=1)
                    if "SMA 2" in indicators_sel:
                        sma2_c = technical.calculate_sma(comp_data_plot, sma2_val)
                        fig.add_trace(go.Scatter(x=common_idx, y=sma2_c, mode='lines', name=f"SMA {sma2_val} ({compare_name})", line=dict(color='blue', width=1, dash='dot')), row=1, col=1)

        fig.update_layout(
            template="plotly_dark", 
            height=700 if has_rsi or is_pro_mode else 500,
            xaxis_rangeslider_visible=False
        )
        
        if is_log:
            fig.update_yaxes(type="log", row=1, col=1)
            
        st.plotly_chart(fig, use_container_width=True)

        if not y_data_filtered.empty:
            metrics_returns = transformations.calculate_log_returns(y_data_filtered).replace([np.inf, -np.inf], np.nan).dropna()
            ret_ann = portfolio.calculate_cagr(y_data_filtered)
            vola = to_scalar(risk.calculate_annualized_volatility(metrics_returns))
            sharpe = to_scalar(portfolio.calculate_sharpe_ratio(metrics_returns, risk_free_rate=risk_free)) 
            var_95 = to_scalar(risk.calculate_value_at_risk(metrics_returns))
            mdd = to_scalar(risk.calculate_max_drawdown(y_data_filtered))
            
            start_v = to_scalar(y_data_filtered.iloc[0])
            end_v = to_scalar(y_data_filtered.iloc[-1])
            total_ret = (end_v / start_v) - 1 if start_v != 0 else 0

            irr_val = 0.0
            if mode == "Portfolio Manager" and time_range_mode == "Maximal / Portfolio-Start":
                 if st.session_state.current_pf_data:
                    cf_list = []
                    for t, d in st.session_state.current_pf_data.items():
                        b_date = pd.to_datetime(d.get('buy_date', datetime.today()))
                        inv_amt = d.get('invested', 0.0)
                        cf_list.append((b_date, -inv_amt))
                    cf_list.append((datetime.today(), total_val))
                    irr_val = portfolio.calculate_xirr(cf_list)
            else:
                irr_val = ret_ann

            st.markdown("#### Performance (im Zeitraum)")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Gesamtrendite", f"{total_ret:+.2%}", help="Absolut im Zeitraum")
            m2.metric("Rendite p.a. (CAGR)", f"{ret_ann:.2%}", help="Geometrischer Durchschnitt")
            if mode == "Portfolio Manager" and time_range_mode == "Maximal / Portfolio-Start":
                m3.metric("Interner Zinsfuss (IRR)", f"{irr_val:.2%}", help="Geldgewichtet (Gesamt)")
            else:
                m3.metric("Rendite p.a.", f"{ret_ann:.2%}", help="Im gewählten Fenster (Zeitgewichtet)")
            m4.metric("Sharpe Ratio", f"{sharpe:.2f}")

            st.markdown("#### Risiko (im Zeitraum)")
            r1, r2, r3 = st.columns(3)
            r1.metric("Max Drawdown", f"{mdd:.2%}")
            r2.metric("Vola p.a.", f"{vola:.2%}")
            r3.metric("VaR 95", f"{var_95:.2%}")

            if news_sources:
                st.markdown("---")
                render_news_section(news_sources)

            if show_advanced_stats:
                st.markdown("---")
                
                st.subheader("Rendite-Verteilung (im Zeitraum)")
                fig_hist = go.Figure()
                
                clean_series = metrics_returns
                
                if len(clean_series) > 2:
                    fig_hist.add_trace(go.Histogram(
                        x=clean_series,
                        nbinsx=50,
                        name="Renditen",
                        histnorm='probability density',
                        marker_color='#1f77b4',
                        opacity=0.75
                    ))
                    
                    try:
                        mu, std = norm.fit(clean_series)
                        xmin = min(clean_series)
                        xmax = max(clean_series)
                        x_range = np.linspace(xmin, xmax, 100)
                        p = norm.pdf(x_range, mu, std)
                        
                        fig_hist.add_trace(go.Scatter(
                            x=x_range, y=p,
                            mode='lines',
                            name='Normalverteilung',
                            line=dict(color='red', width=2)
                        ))
                    except:
                        pass

                    fig_hist.update_layout(
                        template="plotly_dark",
                        height=350,
                        xaxis_title="Tägliche Log-Rendite",
                        yaxis_title="Dichte",
                        showlegend=True,
                        margin=dict(t=30, b=20)
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                else:
                    st.caption("Zu wenig Daten fuer Verteilung.")

                st.markdown("---")
                st.subheader("Rolling Volatility")
                vola_window = st.slider("Zeitfenster (Tage)", min_value=10, max_value=250, value=30, key="volawin")
                
                rolling_vola = main_series.rolling(window=vola_window).std() * np.sqrt(252)
                
                if display_start:
                    rolling_vola = rolling_vola[rolling_vola.index >= display_start]
                if display_end:
                    rolling_vola = rolling_vola[rolling_vola.index <= display_end]

                fig_vola = go.Figure()
                fig_vola.add_trace(go.Scatter(
                    x=rolling_vola.index,
                    y=rolling_vola,
                    mode='lines',
                    name=f'{vola_window}T Vola p.a.',
                    line=dict(color='#ff7f0e', width=1.5)
                ))
                fig_vola.update_layout(
                    template="plotly_dark",
                    height=350,
                    yaxis_title="Volatilität",
                    xaxis_title="",
                    showlegend=True,
                    margin=dict(t=30, b=20),
                    yaxis=dict(tickformat=".0%")
                )
                st.plotly_chart(fig_vola, use_container_width=True)

            if show_options:
                st.markdown("---")
                curr_price = to_scalar(y_data_filtered.iloc[-1]) if not y_data_filtered.empty else None
                
                if mode == "Einzel-Analyse":
                    render_options_analysis(main_name, curr_price, risk_free)
                
                elif mode == "Vergleich":
                    st.subheader("Options-Daten")
                    opt_tab1, opt_tab2 = st.tabs([f"{main_name}", f"{compare_name}"])
                    
                    with opt_tab1:
                        render_options_analysis(main_name, curr_price, risk_free)
                    with opt_tab2:
                        curr_price_2 = to_scalar(compare_prices.iloc[-1]) if compare_prices is not None else None
                        render_options_analysis(compare_name, curr_price_2, risk_free)

if 'logged_in_user' not in st.session_state:
    st.session_state.logged_in_user = None

if st.session_state.logged_in_user is None:
    st.title("Willkommen bei Vektor")
    tab1, tab2 = st.tabs(["Anmelden", "Registrieren"])
    with tab1:
        l_user = st.text_input("Name", key="l_u").strip()
        l_pw = st.text_input("Passwort", type="password", key="l_p").strip()
        if st.button("Login"):
            res = check_login(l_user, l_pw)
            if res == "OK": 
                st.session_state.logged_in_user = l_user
                st.rerun()
            elif res == "NOT_APPROVED":
                st.warning("Warte auf Admin.")
            else:
                st.error("Fehler.")
    with tab2:
        r_user = st.text_input("Name", key="r_u").strip()
        r_pw = st.text_input("Passwort", type="password", key="r_p").strip()
        if st.button("Register"):
            if create_user(r_user, r_pw):
                st.success("Registriert! Warte auf Freigabe.")
            else:
                st.error("Vergeben.")
else:
    run_dashboard(st.session_state.logged_in_user)