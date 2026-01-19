import yfinance as yf
import requests
import streamlit as st
import pandas as pd
import xml.etree.ElementTree as ET
import email.utils
import urllib.parse
import re

REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

@st.cache_data(ttl=15)
def get_stock_data(ticker, period="max", interval="1d"):
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except Exception:
        return None

@st.cache_data(ttl=3600)
def get_company_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        return stock.info
    except Exception:
        return {}

@st.cache_data(ttl=86400)
def get_asset_currency(ticker):
    info = get_company_info(ticker)
    return info.get('currency', 'USD')

@st.cache_data(ttl=3600)
def get_exchange_rate_series(source_currency, target_currency, period="max"):
    if source_currency == target_currency:
        return None
    pair_direct = f"{source_currency}{target_currency}=X"
    data = yf.download(pair_direct, period=period, progress=False)
    if not data.empty:
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data['Close']
    pair_inverse = f"{target_currency}{source_currency}=X"
    data_inv = yf.download(pair_inverse, period=period, progress=False)
    if not data_inv.empty:
        if isinstance(data_inv.columns, pd.MultiIndex):
            data_inv.columns = data_inv.columns.get_level_values(0)
        return 1.0 / data_inv['Close']
    return None

@st.cache_data(ttl=3600)
def get_central_bank_rate(currency="EUR"):
    if currency == "USD":
        try:
            df = yf.download("^IRX", period="5d", progress=False)
            if not df.empty:
                val = df['Close'].iloc[-1]
                if isinstance(val, pd.Series): val = val.iloc[0]
                return float(val) / 100
        except: pass
    rates = {"EUR": 0.0200, "USD": 0.0450, "GBP": 0.0400, "CHF": 0.0000}
    return rates.get(currency, 0.02)

@st.cache_data(ttl=86400)
def get_cumulative_split_factor(ticker, buy_date_str):
    try:
        stock = yf.Ticker(ticker)
        splits = stock.splits
        if splits.empty: return 1.0
        buy_date = pd.to_datetime(buy_date_str).tz_localize(None)
        relevant_splits = splits[splits.index.tz_localize(None) > buy_date]
        if relevant_splits.empty: return 1.0
        factor = 1.0
        for split_val in relevant_splits:
            if split_val > 0: factor *= split_val
        return factor
    except Exception:
        return 1.0

@st.cache_data(ttl=300)
def get_latest_news(ticker):
    print(f"Lade News fuer {ticker}")
    
    search_term = ticker
    try:
        info = get_company_info(ticker)
        raw_name = info.get('longName') or info.get('shortName')
        
        if raw_name:
            clean_name = re.sub(r'\b(AG|SE|GmbH|Co\.?|KG|Corp\.?|Inc\.?|Ltd\.?|PLC|S\.A\.)\b', '', raw_name, flags=re.IGNORECASE)
            clean_name = clean_name.split('-')[0].split(',')[0]
            search_term = clean_name.strip()
            print(f"{ticker} -> Suche nach '{search_term}'")
    except:
        pass

    try:
        if ticker.endswith(".DE") or ticker.endswith(".F") or "EUR" in ticker:
            hl, gl, ceid = "de", "DE", "DE:de"
        else:
            hl, gl, ceid = "en-US", "US", "US:en"
            
        encoded_query = urllib.parse.quote(search_term)
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}+when:7d&hl={hl}&gl={gl}&ceid={ceid}"
        
        resp = requests.get(rss_url, headers=REQUEST_HEADERS, timeout=5)
        
        if resp.status_code == 200:
            root = ET.fromstring(resp.content)
            news_list = []
            
            for item in root.findall('.//item')[:10]:
                title = item.find('title').text
                link = item.find('link').text
                pub_date = item.find('pubDate').text
                source = item.find('source').text if item.find('source') is not None else "Google News"
                
                ts = 0
                if pub_date:
                    try:
                        ts = int(email.utils.parsedate_to_datetime(pub_date).timestamp())
                    except: ts = 0
                
                news_list.append({
                    'title': title,
                    'link': link,
                    'publisher': source,
                    'providerPublishTime': ts,
                    'thumbnail': None
                })
            
            if news_list:
                print(f"{ticker}: {len(news_list)} News via Google gefunden.")
                return news_list
    except Exception as e:
        print(f"{ticker}: Google News Fehler: {e}")

    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={search_term}"
        resp = requests.get(url, headers=REQUEST_HEADERS, timeout=5)
        data = resp.json()
        if 'news' in data and len(data['news']) > 0:
            print(f"{ticker}: Fallback via Yahoo Search")
            return data['news']
    except: pass

    return []

def search_assets(query):
    if not query: return []
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=10&newsCount=0"
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=5)
        data = response.json()
        results = []
        if 'quotes' in data:
            for item in data['quotes']:
                if 'symbol' in item:
                    symbol = item['symbol']
                    name = item.get('shortname', item.get('longname', symbol))
                    exch = item.get('exchDisp', item.get('exchange', '???'))
                    kind = item.get('typeDisp', item.get('quoteType', 'Asset'))
                    label = f"{name} ({symbol}) | {kind} | {exch}"
                    results.append(label)
        return results
    except Exception:
        return []
    

@st.cache_data(ttl=300) 
def get_option_expirations(ticker):
    try:
        stock = yf.Ticker(ticker)
        return stock.options
    except Exception:
        return ()

@st.cache_data(ttl=60) 
def get_option_chain(ticker, date):
    try:
        stock = yf.Ticker(ticker)
        chain = stock.option_chain(date)
        return chain.calls, chain.puts
    except Exception:
        return None, None