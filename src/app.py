import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Poker Analysis Dashboard", layout="wide")

st.title("Poker Analysis Dashboard")

outdir = st.sidebar.text_input("Output directory", value="poker_out")

try:
    summary = pd.read_csv(f"{outdir}/player_summary.csv")
    sizing = pd.read_csv(f"{outdir}/sizing_by_wetness.csv")
    events = pd.read_parquet(f"{outdir}/events.parquet")
except Exception as e:
    st.error(f"Could not load outputs from '{outdir}'. Run run_analysis.py first.\n\nError: {e}")
    st.stop()

players = summary["player"].tolist()
player = st.sidebar.selectbox("Player", players)

st.subheader("Player Summary")
st.dataframe(summary[summary["player"] == player], use_container_width=True)

st.subheader("Aggression Frequencies (bet/raise per opportunity)")
row = summary[summary["player"] == player].iloc[0]
cols = ["aggr_freq_PREFLOP", "aggr_freq_FLOP", "aggr_freq_TURN", "aggr_freq_RIVER"]
st.write({c: float(row[c]) if pd.notna(row[c]) else None for c in cols})

st.subheader("Sizing distribution by wetness")
sub = sizing[sizing["player"] == player].copy()

wet = st.selectbox("Wetness bin", ["PREFLOP", "dry", "medium", "wet"])
street = st.selectbox("Street", ["PREFLOP", "FLOP", "TURN", "RIVER"])

sub2 = sub[(sub["wetness_bin"] == wet) & (sub["street"] == street)]
st.dataframe(sub2.sort_values("pct_within_context", ascending=False), use_container_width=True)

st.subheader("Recent actions (debug)")
recent = events[(events["player"] == player) & (events["kind"].isin(["bet", "raise", "call", "fold", "check"]))].copy()
recent = recent.sort_values("idx", ascending=False).head(50)
st.dataframe(recent[["hand_no", "street", "kind", "amount", "delta_put_in", "pot_before", "pot_after", "wetness", "raw"]], use_container_width=True)
