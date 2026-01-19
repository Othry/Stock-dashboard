import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from scipy.stats import norm
from data.loader import search_assets, get_latest_news, get_option_expirations, get_option_chain, get_asset_currency
from analysis.greeks import calculate_greeks, black_scholes_price

def to_scalar(val):
    if isinstance(val, (pd.Series, pd.DataFrame, np.ndarray)):
        try:
            if isinstance(val, pd.DataFrame):
                return float(val.iloc[0, 0])
            if len(val) > 0:
                return float(val.iloc[0]) if hasattr(val, 'iloc') else float(val[0])
            return 0.0
        except:
            return 0.0
    return float(val)

def get_price_at_date(df, date_str):
    try:
        target_date = pd.to_datetime(date_str)
        past_data = df.loc[:target_date]
        if past_data.empty:
            return None 
        return to_scalar(past_data.iloc[-1])
    except:
        return None

def asset_selector(label, key_prefix, default_symbol="AAPL", predefined_options=None):
    options = ["Suche..."]
    if predefined_options:
        options = list(predefined_options.keys()) + options
    
    if not predefined_options:
        selection_mode = "Suche..."
    else:
        selection_mode = st.selectbox(f"{label}", options, key=f"{key_prefix}_mode")

    selected_ticker = None

    if selection_mode == "Suche...":
        search_label = f"{label} suchen" if not predefined_options else "Suche starten"
        query = st.text_input(search_label, key=f"{key_prefix}_search")
        if query:
            results = search_assets(query)
            if results:
                selection = st.selectbox("Ergebnisse", results, key=f"{key_prefix}_results")
                if "(" in selection and ")" in selection:
                    selected_ticker = selection.split('(')[1].split(')')[0]
                else:
                    selected_ticker = selection
            else:
                st.warning("Nichts gefunden.")
        else:
            if not predefined_options:
                selected_ticker = default_symbol
    elif predefined_options:
        selected_ticker = predefined_options[selection_mode]
    return selected_ticker

def _show_news_for_ticker(ticker):
    news_items = get_latest_news(ticker)
    valid_news = [n for n in news_items if n.get('title')]
    
    if not valid_news:
        st.info(f"Keine aktuellen Nachrichten für {ticker}.")
        return

    for item in valid_news[:5]:
        title = item.get('title')
        publisher = item.get('publisher', 'Quelle unbekannt')
        link = item.get('link', '#')
        ts = item.get('providerPublishTime', 0)
        date_str = ""
        if ts and ts > 0:
            try:
                date_str = datetime.fromtimestamp(ts).strftime('%d.%m.%Y %H:%M')
            except:
                date_str = ""
        
        with st.expander(f"{title}"):
            c1, c2 = st.columns([3, 1])
            c1.markdown(f"**{publisher}** | {date_str}")
            c1.markdown(f"[Artikel lesen]({link})")
            thumbnail = item.get('thumbnail')
            if thumbnail and 'resolutions' in thumbnail:
                try:
                    c2.image(thumbnail['resolutions'][0]['url'], width=100)
                except: pass

def render_news_section(tickers_map):
    if not tickers_map: return
    c1, c2 = st.columns([5, 1])
    c1.markdown("### News-Feed")
    if c2.button("Aktualisieren"):
        get_latest_news.clear()
        st.rerun()
    
    if isinstance(tickers_map, list): tickers_map = {t: t for t in tickers_map}
    ticker_keys = list(tickers_map.keys())
    
    if len(ticker_keys) > 1:
        tabs = st.tabs([f"{tickers_map[t]}" for t in ticker_keys])
        for i, ticker in enumerate(ticker_keys):
            with tabs[i]:
                _show_news_for_ticker(ticker)
    else:
        t = ticker_keys[0]
        st.caption(f"News für {tickers_map[t]} ({t})")
        _show_news_for_ticker(t)

def render_straddle_builder(ticker, current_price, risk_free_rate=0.04):
    if current_price is None or current_price == 0:
        st.error("Aktueller Preis nicht verfügbar.")
        return

    expirations = get_option_expirations(ticker)
    if not expirations:
        st.warning("Keine Optionsdaten verfügbar.")
        return

    asset_currency = get_asset_currency(ticker)

    if f"strike_put_{ticker}" not in st.session_state:
        st.session_state[f"strike_put_{ticker}"] = None
    if f"strike_call_{ticker}" not in st.session_state:
        st.session_state[f"strike_call_{ticker}"] = None

    c_date, c_iv, c_atm = st.columns([2, 2, 1])
    selected_date = c_date.selectbox(f"Verfallsdatum", expirations, key=f"strad_date_{ticker}")
    
    iv_adjustment = c_iv.slider("IV Simulation (%)", min_value=-50, max_value=50, value=0, step=1, format="%d%%") / 100.0
    
    calls, puts = get_option_chain(ticker, selected_date)
    
    if calls is not None and puts is not None:
        calls = calls[['strike', 'ask', 'bid', 'lastPrice', 'volume', 'impliedVolatility']].copy()
        puts = puts[['strike', 'ask', 'bid', 'lastPrice', 'volume', 'impliedVolatility']].copy()
    
        exp_date = pd.to_datetime(selected_date)
        if exp_date.tz is not None:
            exp_date = exp_date.tz_localize(None)
            
        today = pd.Timestamp.now()
        if today.tz is not None:
            today = today.tz_localize(None)
            
        time_diff = exp_date - today
        days_to_maturity = time_diff.total_seconds() / (24 * 3600)
    
        T = max(days_to_maturity / 365.0, 0.0027)

        def get_robust_price(row, option_type):
            bid = row['bid']
            ask = row['ask']
            last = row['lastPrice']
            
            if bid > 0 and ask > 0:
                price_est = (bid + ask) / 2
            elif last > 0:
                price_est = last
            else:
                price_est = ask if ask > 0 else 0.01

            intrinsic = 0.0
            if option_type == "call":
                intrinsic = max(0, current_price - row['strike'])
            else:
                intrinsic = max(0, row['strike'] - current_price)
            
            return max(price_est, intrinsic)

        calls['price'] = calls.apply(lambda row: get_robust_price(row, "call"), axis=1)
        puts['price'] = puts.apply(lambda row: get_robust_price(row, "put"), axis=1)
        
        common_strikes = sorted(list(set(calls['strike']).intersection(set(puts['strike']))))
        
        if not common_strikes:
            st.warning("Keine passenden Strikes gefunden.")
            return

        closest_atm_strike = min(common_strikes, key=lambda x: abs(x - current_price))
        
        if c_atm.button("Set ATM"):
            st.session_state[f"strike_put_{ticker}"] = closest_atm_strike
            st.session_state[f"strike_call_{ticker}"] = closest_atm_strike
            st.rerun()

        if st.session_state[f"strike_put_{ticker}"] is None:
             st.session_state[f"strike_put_{ticker}"] = closest_atm_strike
        if st.session_state[f"strike_call_{ticker}"] is None:
             st.session_state[f"strike_call_{ticker}"] = closest_atm_strike
        
        try:
            idx_put = common_strikes.index(st.session_state[f"strike_put_{ticker}"])
        except: idx_put = common_strikes.index(closest_atm_strike)
        
        try:
            idx_call = common_strikes.index(st.session_state[f"strike_call_{ticker}"])
        except: idx_call = common_strikes.index(closest_atm_strike)

        col_put, col_call = st.columns(2)
        strike_put = col_put.selectbox("Strike PUT", common_strikes, index=idx_put, key=f"sel_put_{ticker}")
        strike_call = col_call.selectbox("Strike CALL", common_strikes, index=idx_call, key=f"sel_call_{ticker}")
        
        st.session_state[f"strike_put_{ticker}"] = strike_put
        st.session_state[f"strike_call_{ticker}"] = strike_call

        try:
            put_leg = puts[puts['strike'] == strike_put].iloc[0]
            call_leg = calls[calls['strike'] == strike_call].iloc[0]
        except:
            st.error("Datenfehler.")
            return

        put_cost = put_leg['price']
        call_cost = call_leg['price']
        total_cost = put_cost + call_cost
        
        raw_iv_put = put_leg['impliedVolatility']
        raw_iv_call = call_leg['impliedVolatility']
        
        if raw_iv_put < 0.001: raw_iv_put = raw_iv_call
        if raw_iv_call < 0.001: raw_iv_call = raw_iv_put
        
        if raw_iv_put < 0.001 and raw_iv_call < 0.001:
            base_iv = 0.30
        else:
            base_iv = (raw_iv_put + raw_iv_call) / 2
            
        avg_iv = base_iv + iv_adjustment
        
        if avg_iv > 5.0:
            avg_iv = avg_iv / 100.0
        
        be_lower = strike_put - total_cost
        be_upper = strike_call + total_cost

        try:
            sigma_calc = float(avg_iv)
            if sigma_calc > 1.0: 
                sigma_calc /= 100.0
            
            vol_term = sigma_calc * np.sqrt(T)
            
            drift_term = (risk_free_rate - 0.5 * sigma_calc**2) * T
            
            log_ret_lower = np.log(be_lower / current_price)
            log_ret_upper = np.log(be_upper / current_price)
            
            z_lower = (log_ret_lower - drift_term) / vol_term
            z_upper = (log_ret_upper - drift_term) / vol_term
            
            prob_below = norm.cdf(z_lower)
            prob_above = 1.0 - norm.cdf(z_upper)
            
            pop = prob_below + prob_above
        except:
            pop = 0.0

        p_delta, pg, pt, pv, pr = calculate_greeks(current_price, strike_put, T, risk_free_rate, avg_iv, "put")
        cd, cg, ct, cv, cr = calculate_greeks(current_price, strike_call, T, risk_free_rate, avg_iv, "call")
        
        net_delta = p_delta + cd
        net_gamma = pg + cg
        net_theta = pt + ct
        net_vega = pv + cv

        st.markdown(f"#### Analyse & Risiko ({asset_currency})")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Kosten (Risk)", f"{total_cost:.2f}")
        m2.metric("Breakeven", f"{be_lower:.2f} / {be_upper:.2f}")
        m3.metric("Prob. of Profit", f"{pop:.1%}", help="Wahrscheinlichkeit, dass Kurs außerhalb der BEs endet")
        m4.metric("Net Delta", f"{net_delta:.2f}", delta_color="off", help="Positiv = Bullisch, Negativ = Bearish")
        
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Net Vega", f"{net_vega:.2f}", help="P&L Änderung bei 1% Vola Anstieg")
        g2.metric("Net Theta", f"{net_theta:.2f}", help="Täglicher Zeitwertverlust")
        g3.metric("Net Gamma", f"{net_gamma:.3f}")
        g4.metric("Implied Volatility", f"{base_iv:.1%}", delta=f"{iv_adjustment:+.0%}", help=f"Durchschnitt IV (Put: {raw_iv_put:.1%} / Call: {raw_iv_call:.1%})")

        range_min = current_price * 0.5
        range_max = current_price * 1.5
        x_prices = np.linspace(range_min, range_max, 200)
        
        y_pnl_exp = (np.maximum(strike_put - x_prices, 0) + np.maximum(x_prices - strike_call, 0)) - total_cost

        bs_put_vals = black_scholes_price(x_prices, strike_put, T, risk_free_rate, avg_iv, "put")
        bs_call_vals = black_scholes_price(x_prices, strike_call, T, risk_free_rate, avg_iv, "call")
        y_pnl_today = (bs_put_vals + bs_call_vals) - total_cost

        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=x_prices, y=y_pnl_exp, mode='lines', name='Verfall (T=0)', line=dict(color='gray', width=2, dash='dash')))
        
        fig.add_trace(go.Scatter(x=x_prices, y=y_pnl_today, mode='lines', name='Heute (T=Aktuell)', line=dict(color='#00CC96', width=3)))

        fig.add_hline(y=0, line_color="white", line_dash="dot")
        
        fig.add_vline(x=current_price, line_color="yellow", line_width=1, annotation_text="Spot")
        fig.add_vline(x=be_lower, line_color="red", line_dash="dot", opacity=0.5)
        fig.add_vline(x=be_upper, line_color="red", line_dash="dot", opacity=0.5)

        sigma_sqrt_t = avg_iv * np.sqrt(T) 
        
        drift_term = (risk_free_rate - 0.5 * avg_iv**2) * T
        
        s_1std_upper = current_price * np.exp(drift_term + sigma_sqrt_t)
        s_1std_lower = current_price * np.exp(drift_term - sigma_sqrt_t)
        
        fig.add_vrect(x0=s_1std_lower, x1=s_1std_upper, 
                      fillcolor="white", opacity=0.08, 
                      annotation_text="1σ Range", annotation_position="top left")

        fig.update_layout(
            title=f"P&L Simulation (Basis IV: {base_iv:.1%})",
            xaxis_title=f"Kurs ({asset_currency})",
            yaxis_title=f"P&L ({asset_currency})",
            template="plotly_dark",
            height=450,
            margin=dict(t=30, b=20),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("Fehler beim Laden der Daten.")

def render_options_analysis(ticker, current_price_ref=None, risk_free_rate=0.04):
    st.subheader(f"Options-Kette: {ticker}")
    
    asset_currency = get_asset_currency(ticker)
    
    show_builder = st.checkbox("Straddle / Strangle Builder anzeigen", value=False)
    
    if show_builder:
        with st.container(border=True):
            render_straddle_builder(ticker, current_price_ref)

    st.markdown("---")
    
    st.info(f"Hinweis: Alle Optionspreise, Strikes und Greeks werden in der Originalwährung **{asset_currency}** angezeigt.")

    expirations = get_option_expirations(ticker)
    if not expirations:
        st.warning("Keine Optionsdaten verfügbar.")
        return

    selected_date_str = st.selectbox(f"Verfallsdatum Tabelle ({ticker})", expirations, key=f"opt_date_{ticker}")
    
    calls, puts = get_option_chain(ticker, selected_date_str)
    
    if calls is not None and puts is not None:
        try:
            exp_date = pd.to_datetime(selected_date_str)
            if exp_date.tz is not None:
                exp_date = exp_date.tz_localize(None)
            
            today = pd.Timestamp.now()
            if today.tz is not None:
                today = today.tz_localize(None)
                
            time_diff = exp_date - today
            days_to_maturity = time_diff.total_seconds() / (24 * 3600)
            T = max(days_to_maturity / 365.0, 0.0001)
        except:
            T = 0.01

        tab_c, tab_p = st.tabs(["Calls", "Puts"])
        
        base_cols = ['strike', 'lastPrice', 'bid', 'ask', 'impliedVolatility']
        greek_cols = ['Delta', 'Gamma', 'Theta', 'Vega']
        final_cols = base_cols + greek_cols + ['volume', 'openInterest']
        
        fmt = {
            'strike': '{:.2f}', 'lastPrice': '{:.2f}', 'bid': '{:.2f}', 'ask': '{:.2f}', 
            'impliedVolatility': '{:.2%}',
            'Delta': '{:.3f}', 'Gamma': '{:.4f}', 'Theta': '{:.4f}', 'Vega': '{:.4f}'
        }

        with tab_c:
            if current_price_ref:
                calls = calls.copy()
                d, g, t, v, r = calculate_greeks(
                    current_price_ref, calls['strike'], T, risk_free_rate, calls['impliedVolatility'], "call"
                )
                calls['Delta'] = d
                calls['Gamma'] = g
                calls['Theta'] = t
                calls['Vega'] = v
                
                available_cols = [c for c in final_cols if c in calls.columns]
                styled_calls = calls[available_cols].style.format(fmt)
                st.dataframe(styled_calls, use_container_width=True, height=500)
            else:
                st.dataframe(calls, use_container_width=True)
            
        with tab_p:
            if current_price_ref:
                puts = puts.copy()
                p_delta, g, t, v, r = calculate_greeks(
                    current_price_ref, puts['strike'], T, risk_free_rate, puts['impliedVolatility'], "put"
                )
                puts['Delta'] = p_delta
                puts['Gamma'] = g
                puts['Theta'] = t
                puts['Vega'] = v
                
                available_cols = [c for c in final_cols if c in puts.columns]
                styled_puts = puts[available_cols].style.format(fmt)
                st.dataframe(styled_puts, use_container_width=True, height=500)
            else:
                st.dataframe(puts, use_container_width=True)
    else:
        st.error("Fehler beim Laden der Kette.")