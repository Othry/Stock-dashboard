import streamlit as st
import pandas as pd
import json
import os
import numpy as np
import plotly.graph_objects as go

STRATEGY_FILE = "strategies.json"

# --- 1. Daten-Management (CRUD) ---

def load_strategies():
    if not os.path.exists(STRATEGY_FILE):
        return {}
    try:
        with open(STRATEGY_FILE, "r") as f:
            return json.load(f)
    except:
        return {}

def save_strategy(name, legs, exit_rules):
    strategies = load_strategies()
    strategies[name] = {
        "legs": legs,
        "exit_rules": exit_rules
    }
    with open(STRATEGY_FILE, "w") as f:
        json.dump(strategies, f, indent=4)

def delete_strategy(name):
    strategies = load_strategies()
    if name in strategies:
        del strategies[name]
        with open(STRATEGY_FILE, "w") as f:
            json.dump(strategies, f, indent=4)

# --- 2. Visuelle Vorschau (Payoff Diagramm Form) ---
def plot_strategy_shape(legs):
    """
    Zeigt die Form der Strategie (z.B. Zelt für Iron Condor)
    basierend auf einem fiktiven Spot-Preis von 100.
    """
    spot = 100
    x = np.linspace(70, 130, 200)
    y_total = np.zeros_like(x)
    
    for leg in legs:
        # Wir simulieren Strikes basierend auf der Definition
        strike = 100 # Default ATM
        
        if leg['strike_method'] == "Pct of Spot":
            # z.B. 105% -> Strike 105
            strike = spot * (leg['strike_val'] / 100.0)
        elif leg['strike_method'] == "Delta":
            # Grobe Schätzung für Visualisierung: Delta 50 = ATM, Delta 20 = OTM
            # Call: Delta 0.5 -> 100, Delta 0.2 -> 110
            # Put: Delta -0.5 -> 100, Delta -0.2 -> 90
            # (Dies ist nur für die Form-Anzeige, keine exakte Mathe!)
            if leg['type'] == "call":
                strike = spot + (0.5 - leg['strike_val']) * 40 
            else:
                strike = spot - (abs(leg['strike_val']) - 0.5) * 40
        
        # Payoff berechnen
        sign = 1 if leg['action'] == "buy" else -1
        
        if leg['type'] == "call":
            y = np.maximum(x - strike, 0) * sign
        else:
            y = np.maximum(strike - x, 0) * sign
            
        y_total += y

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y_total, mode='lines', name='Payoff Shape', line=dict(color='cyan', width=3)))
    fig.add_hline(y=0, line_color="white", line_dash="dot")
    fig.update_layout(
        title="Strategie Profil (Abstrakt)",
        xaxis_title="Relativer Kurs (Spot=100)",
        yaxis_title="Payoff Struktur",
        template="plotly_dark",
        height=300,
        margin=dict(t=30, b=20)
    )
    return fig

# --- 3. Der UI Builder ---

def render_strategy_builder():
    st.subheader("Strategie Baukasten")
    
    # Session State für den aktuellen Bauprozess
    if "builder_legs" not in st.session_state:
        st.session_state.builder_legs = []
    
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        st.markdown("##### Neues Leg hinzufügen")
        l_type = st.selectbox("Option Typ", ["call", "put"], key="b_type")
        l_action = st.selectbox("Aktion", ["buy", "sell"], key="b_action")
        
        # Backtest-Methode: Wie finden wir den Strike in der Vergangenheit?
        l_method = st.selectbox("Strike Wahl", ["Delta", "Pct of Spot"], key="b_method", 
                                help="Delta: Nutzt Greeks (Profi). Pct: Nutzt % vom Aktienkurs.")
        
        if l_method == "Delta":
            l_val = st.slider("Delta Ziel", 0.05, 0.95, 0.50, step=0.05, 
                              help="0.50 = ATM, 0.20 = OTM")
        else:
            l_val = st.number_input("Prozent vom Spot", 80.0, 120.0, 100.0, step=1.0,
                                    help="100 = ATM, 110 = 10% OTM (Call)")

        if st.button("Leg hinzufügen"):
            st.session_state.builder_legs.append({
                "type": l_type,
                "action": l_action,
                "strike_method": l_method,
                "strike_val": l_val
            })
            st.rerun()

        if st.button("Alle Legs löschen"):
            st.session_state.builder_legs = []
            st.rerun()

    with col_right:
        st.markdown("##### Aktuelle Konstruktion")
        
        if st.session_state.builder_legs:
            # Tabelle der Legs
            df_legs = pd.DataFrame(st.session_state.builder_legs)
            st.dataframe(df_legs, use_container_width=True, hide_index=True)
            
            # Visualisierung
            st.plotly_chart(plot_strategy_shape(st.session_state.builder_legs), use_container_width=True)
            
            st.markdown("---")
            st.markdown("##### Speicher-Optionen")
            
            c1, c2, c3 = st.columns(3)
            strat_name = c1.text_input("Strategie Name", placeholder="z.B. Iron Condor")
            
            # Exit Regeln (Simple)
            profit_target = c2.number_input("Take Profit (%)", value=50, step=10)
            stop_loss = c3.number_input("Stop Loss (%)", value=100, step=10)
            
            if st.button("Strategie Speichern", type="primary"):
                if strat_name:
                    exit_rules = {"take_profit": profit_target, "stop_loss": stop_loss}
                    save_strategy(strat_name, st.session_state.builder_legs, exit_rules)
                    st.success(f"Strategie '{strat_name}' gespeichert!")
                    st.session_state.builder_legs = [] # Reset
                    st.rerun()
                else:
                    st.error("Bitte Namen eingeben.")
        else:
            st.info("Füge Legs hinzu, um eine Strategie zu bauen.")

    # --- Liste der gespeicherten Strategien ---
    st.markdown("---")
    st.subheader("Gespeicherte Strategien")
    saved = load_strategies()
    if saved:
        for name, data in saved.items():
            with st.expander(f"{name} ({len(data['legs'])} Legs)"):
                st.write("**Exit Regeln:**", data['exit_rules'])
                st.table(pd.DataFrame(data['legs']))
                if st.button(f"Löschen {name}", key=f"del_{name}"):
                    delete_strategy(name)
                    st.rerun()
    else:
        st.caption("Noch keine Strategien gespeichert.")