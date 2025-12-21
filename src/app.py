import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Poker Analysis Dashboard", layout="wide")



st.title("Poker Analysis Dashboard")

outdir = st.sidebar.text_input("Output directory", value="outputs")

try:
    summary = pd.read_csv(f"{outdir}/player_summary.csv")
    sizing = pd.read_csv(f"{outdir}/sizing_by_wetness.csv")
    events = pd.read_parquet(f"{outdir}/events.parquet")
    ranges = pd.read_csv(f"{outdir}/ranges.csv")
    hp = pd.read_csv(f"{outdir}/hand_player_details.csv")

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


st.subheader("Observed Range (Showdown-based)")
pr = ranges[ranges["player"] == player].copy()

st.caption("Range is built from hands where: (1) showdown happened, (2) player voluntarily continued preflop (>= BB), (3) player showed cards.")

st.dataframe(pr, use_container_width=True)

combo = st.selectbox("Select a combo to drill down", ["(pick one)"] + pr["combo"].tolist())

if combo != "(pick one)":
    st.subheader(f"Hands where {player} showed {combo}")

    sub = hp[(hp["player"] == player) & (hp["combo"] == combo)].copy()
    sub = sub.sort_values("hand_no", ascending=False)

    st.dataframe(sub[[
        "hand_no", "voluntary_preflop", "saw_FLOP", "saw_TURN", "saw_RIVER",
        "showdown_happened", "won_hand", "collected_amt", "action_summary"
    ]], use_container_width=True)

    hand_list = sub["hand_no"].astype(int).tolist()
    hand_no = st.selectbox("Select a hand to view full timeline", hand_list)

    st.subheader(f"Full Hand Timeline: Hand #{hand_no}")
    hand_events = events[events["hand_no"] == hand_no].sort_values("idx").copy()

    # Show board cards events first
    boards = hand_events[hand_events["kind"] == "board"][["street", "raw", "wetness"]]
    if len(boards) > 0:
        st.write("Board runout:")
        st.dataframe(boards, use_container_width=True)

    # Show full action log
    st.write("All events:")
    st.dataframe(hand_events[["idx","street","kind","player","amount","delta_put_in","pot_before","pot_after","raw"]], use_container_width=True)

    # Show only selected player's actions for this hand
    st.write(f"{player}'s actions:")
    p_events = hand_events[hand_events["player"] == player]
    st.dataframe(p_events[["idx","street","kind","amount","delta_put_in","pot_before","pot_after","raw"]], use_container_width=True)
