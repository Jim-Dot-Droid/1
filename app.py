import streamlit as st
import pandas as pd
import numpy as np
import os

# --- File paths ---
HISTORY_FILE = "history.csv"
RESULTS_FILE = "results.csv"
FLAT_FILE = "flat_balance.txt"
FIXED_FILE = "fixed_balance.txt"
MART_FILE = "martingale_balance.txt"

# --- Constants ---
INITIAL_BALANCE = 0.1
FLAT_BET = 0.01
FIXED_BET = 0.02
WINDOWS = [20, 50, 100]
MIN_UNDERS = 14  # Threshold for predicting "Above"

# --- Helpers ---
@st.cache_data
def load_csv(file):
    df = pd.read_csv(file)
    if 'multiplier' in df.columns:
        return df['multiplier'].tolist()
    return df.iloc[:,0].tolist()

def load_history():
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)['multiplier'].tolist()
    return []

def save_history(data):
    pd.DataFrame({'multiplier': data}).to_csv(HISTORY_FILE, index=False)

def load_results():
    if os.path.exists(RESULTS_FILE):
        return pd.read_csv(RESULTS_FILE)
    return pd.DataFrame(columns=['prediction', 'actual', 'correct'])

def save_result(prediction, actual):
    correct = ((prediction == "Above") and actual > 2.0) or ((prediction == "Under") and actual <= 2.0)
    df = load_results()
    df.loc[len(df)] = [prediction, actual, correct]
    df.to_csv(RESULTS_FILE, index=False)
    update_balances(prediction, actual)

# --- Balances ---
def get_balance(file, initial=INITIAL_BALANCE):
    if os.path.exists(file):
        with open(file, "r") as f:
            return float(f.read())
    return initial

def set_balance(file, value):
    with open(file, "w") as f:
        f.write(str(value))

def update_balances(prediction, actual):
    # Flat
    flat = get_balance(FLAT_FILE)
    if prediction == "Above":
        flat += FLAT_BET if actual > 2.0 else -FLAT_BET
    set_balance(FLAT_FILE, flat)

    # Fixed
    fixed = get_balance(FIXED_FILE)
    if prediction == "Above":
        fixed += FIXED_BET if actual > 2.0 else -FIXED_BET
    set_balance(FIXED_FILE, fixed)

    # Martingale
    mart = get_balance(MART_FILE)
    bet = FLAT_BET
    mart_pred = prediction
    mart_actual = actual
    if mart_pred == "Above":
        if mart_actual > 2.0:
            mart += bet
            bet = FLAT_BET
        else:
            mart -= bet
            bet *= 2
    set_balance(MART_FILE, mart)

def reset_balances():
    for f in [FLAT_FILE, FIXED_FILE, MART_FILE, RESULTS_FILE]:
        if os.path.exists(f):
            os.remove(f)

# --- Prediction ---
def predict_from_unders(data, window=20, min_unders=MIN_UNDERS, threshold=2.0):
    if len(data) < window:
        return None, None
    recent = np.array(data[-window:])
    under_count = int(np.sum(recent < threshold))
    if under_count >= min_unders:
        return "Above", under_count
    return "Under", under_count

def normalize_input(value):
    return value / 100 if value > 10 else value

# --- Streamlit App ---
def main():
    st.title("Crash Predictor — 20, 50, 100 Rounds with Flat/Fixed/Martingale")

    # Sidebar
    st.sidebar.header("Settings")
    min_unders = st.sidebar.slider("Min unders to predict 'Above' (last 20)", 10, 20, MIN_UNDERS)

    if "history" not in st.session_state:
        st.session_state.history = load_history()

    # Upload
    uploaded = st.file_uploader("Upload multipliers CSV", type=["csv"])
    if uploaded:
        st.session_state.history = load_csv(uploaded)
        save_history(st.session_state.history)
        st.success(f"Loaded {len(st.session_state.history)} multipliers.")

    # Reset / Clear
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Reset all"):
            st.session_state.history = []
            save_history([])
            reset_balances()
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
            st.success("All reset!")
    with col2:
        if st.button("Clear History"):
            st.session_state.history = []
            save_history([])
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
            st.success("History cleared!")

    # Manual input
    st.subheader("Add Multiplier")
    new_val = st.text_input("Enter multiplier (e.g., 1.87 or 187)")
    if st.button("Add multiplier"):
        try:
            val = float(new_val)
            val = normalize_input(val)
            if "last_prediction" in st.session_state:
                save_result(st.session_state.last_prediction, val)
                del st.session_state.last_prediction
            st.session_state.history.append(val)
            save_history(st.session_state.history)
            st.success(f"Added {val}x")
        except Exception:
            st.error("Invalid input")

    # Predictions
    if st.session_state.history:
        data = st.session_state.history
        st.write(f"History length: {len(data)}")

        for w in WINDOWS:
            st.subheader(f"Prediction (last {w})")
            pred, count = predict_from_unders(data, window=w, min_unders=int(min_unders * w / 20))
            if pred:
                st.session_state.last_prediction = pred
                st.write(f"Prediction: **{pred}** (Under count = {count})")
            else:
                st.write(f"Not enough data yet (need {w} rounds)")

    # Accuracy
    st.subheader("Prediction Accuracy")
    results_df = load_results()
    if not results_df.empty:
        total = len(results_df)
        correct = int(results_df['correct'].sum())
        acc = correct / total if total else 0
        st.metric("Total Predictions", total)
        st.metric("Correct Predictions", correct)
        st.metric("Accuracy Rate", f"{acc:.1%}")
        st.dataframe(results_df[::-1].reset_index(drop=True))
    else:
        st.write("No verified predictions yet.")

    # Balances
    st.subheader("Balances")
    st.metric("Flat Bet", f"{get_balance(FLAT_FILE):.4f} SOL")
    st.metric("Fixed Bet", f"{get_balance(FIXED_FILE):.4f} SOL")
    st.metric("Martingale", f"{get_balance(MART_FILE):.4f} SOL")

    # History table
    st.subheader("Crash History")
    if st.session_state.history:
        df = pd.DataFrame({
            "Round #": range(1, len(st.session_state.history) + 1),
            "Multiplier": st.session_state.history
        })
        st.dataframe(df[::-1].reset_index(drop=True))
    else:
        st.write("No history yet.")

if __name__ == "__main__":
    main()
