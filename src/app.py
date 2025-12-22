# src/app.py
import os
import io
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# -----------------------------
# 13x13 grid helpers
# -----------------------------
RANKS = ["A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2"]

def combo_label(i, j):
    r1, r2 = RANKS[i], RANKS[j]
    if i == j:
        return f"{r1}{r2}"
    if i < j:
        return f"{r1}{r2}s"     # suited above diagonal
    return f"{r2}{r1}o"         # offsuit below diagonal

def draw_range_chart(counts_map, title="Range Grid"):
    """
    counts_map: dict combo -> count (e.g., {"KQo": 5, "T8s": 2})
    Highlights squares where count>0, with intensity by count.
    Always draws the full 13x13 grid.
    """
    max_count = max(counts_map.values()) if counts_map else 1

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 13)
    ax.invert_yaxis()
    ax.axis("off")

    for i in range(13):
        for j in range(13):
            lab = combo_label(i, j)
            c = int(counts_map.get(lab, 0))
            intensity = (c / max_count) if max_count > 0 else 0.0

            # gray background for 0, blue intensity for >0
            face = (0.95, 0.95, 0.95, 1.0) if c == 0 else (0.2, 0.4, 0.9, 0.15 + 0.85 * intensity)

            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=face, edgecolor="black", linewidth=0.5))
            ax.text(j + 0.5, i + 0.5, lab, ha="center", va="center", fontsize=8)

    ax.set_title(title)
    return fig


# -----------------------------
# Streamlit setup
# -----------------------------
st.set_page_config(page_title="Poker Analysis Dashboard", layout="wide")
st.title("Poker Analysis Dashboard")

st.sidebar.header("Data Source")
outdir = st.sidebar.text_input("Output directory", value="outputs")

@st.cache_data(show_spinner=False)
def load_data(outdir: str):
    summary = pd.read_csv(os.path.join(outdir, "player_summary.csv"))
    sizing = pd.read_csv(os.path.join(outdir, "sizing_by_wetness.csv"))
    events = pd.read_parquet(os.path.join(outdir, "events.parquet"))
    hp = pd.read_csv(os.path.join(outdir, "hand_player_details.csv"))
    ranges = pd.read_csv(os.path.join(outdir, "ranges.csv"))
    ranges_by_pos = pd.read_csv(os.path.join(outdir, "ranges_by_position.csv"))
    return summary, sizing, events, hp, ranges, ranges_by_pos

try:
    summary, sizing, events, hp, ranges, ranges_by_pos = load_data(outdir)
except Exception as e:
    st.error(
        f"Could not load outputs from '{outdir}'.\n\n"
        f"Make sure you ran:\n"
        f"  python -m src.run_analysis data/raw/<yourfile>.csv --outdir {outdir} --entry_col entry --autodetect_blinds\n\n"
        f"Error:\n{e}"
    )
    st.stop()

# Basic sanity/debug
with st.expander("Debug: loaded file shapes"):
    st.write({
        "player_summary.csv": summary.shape,
        "sizing_by_wetness.csv": sizing.shape,
        "events.parquet": events.shape,
        "hand_player_details.csv": hp.shape,
        "ranges.csv": ranges.shape,
        "ranges_by_position.csv": ranges_by_pos.shape,
    })

# Player selector
players = summary["player"].astype(str).tolist() if "player" in summary.columns else sorted(events["player"].dropna().astype(str).unique().tolist())
player = st.sidebar.selectbox("Player", players)

# -----------------------------
# Player Summary
# -----------------------------
st.subheader("Player Summary")
if "player" in summary.columns:
    st.dataframe(summary[summary["player"] == player], use_container_width=True)
else:
    st.warning("player_summary.csv has no 'player' column. Check parsing output.")

# VPIP + aggression quick view
st.subheader("Key Stats")
row = summary[summary["player"] == player].iloc[0] if ("player" in summary.columns and len(summary[summary["player"] == player]) > 0) else None
if row is not None:
    keys = []
    for c in ["vpip", "vpip_hands", "hands_dealt_ex_bb",
              "aggr_freq_PREFLOP", "aggr_freq_FLOP", "aggr_freq_TURN", "aggr_freq_RIVER"]:
        if c in summary.columns:
            keys.append((c, row[c]))
    st.write({k: (float(v) if pd.notna(v) else None) for k, v in keys})
else:
    st.write("No summary row for this player.")

# -----------------------------
# Sizing distribution
# -----------------------------
st.subheader("Sizing distribution by wetness")
sub = sizing[sizing["player"] == player].copy() if "player" in sizing.columns else pd.DataFrame()

wet = st.selectbox("Wetness bin", ["PREFLOP", "dry", "medium", "wet"], key="wet_bin")
street = st.selectbox("Street", ["PREFLOP", "FLOP", "TURN", "RIVER"], key="street_bin")

if len(sub):
    sub2 = sub[(sub["wetness_bin"] == wet) & (sub["street"] == street)]
    st.dataframe(sub2.sort_values("pct_within_context", ascending=False), use_container_width=True)
else:
    st.caption("No sizing data found for this player.")

# -----------------------------
# Range Grid (this is the 13x13 matrix)
# -----------------------------
st.subheader("Range Grid (Observed at Showdown)")

if len(ranges_by_pos) == 0:
    st.warning("ranges_by_position.csv is empty. Range-by-position filter will be unavailable.")

pos_options = ["ALL"]
if "position" in ranges_by_pos.columns and len(ranges_by_pos):
    pos_options += sorted([p for p in ranges_by_pos["position"].dropna().astype(str).unique().tolist()])

pos_filter = st.selectbox("Filter by position", pos_options, key="pos_filter")

# Build counts_map for grid
counts_map = {}
if pos_filter == "ALL":
    pr = ranges[ranges["player"] == player].copy() if ("player" in ranges.columns and len(ranges)) else pd.DataFrame()
    if len(pr):
        counts_map = dict(zip(pr["combo"].astype(str), pr["count"].astype(int)))
    fig = draw_range_chart(counts_map, title=f"{player} — Observed Range (All Positions)")
else:
    subp = ranges_by_pos[(ranges_by_pos["player"] == player) & (ranges_by_pos["position"].astype(str) == pos_filter)].copy()
    if len(subp):
        counts_map = dict(zip(subp["combo"].astype(str), subp["count"].astype(int)))
    fig = draw_range_chart(counts_map, title=f"{player} — Observed Range ({pos_filter})")

# IMPORTANT: This is what actually displays the grid
st.pyplot(fig, clear_figure=True)

# -----------------------------
# Drill down: combo -> hands -> timeline
# -----------------------------
st.subheader("Drill Down: Combo → Hands")

combo_list = sorted(counts_map.keys())
combo = st.selectbox("Select a combo", ["(pick one)"] + combo_list, key="combo_pick")

if combo != "(pick one)":
    sub = hp[(hp["player"] == player) & (hp["combo"] == combo)].copy()

    if pos_filter != "ALL" and "position" in sub.columns:
        sub = sub[sub["position"].astype(str) == pos_filter]

    sub = sub.sort_values("hand_no", ascending=False)

    cols = [c for c in ["hand_no", "position", "n_players", "won_hand", "collected_amt", "action_summary"] if c in sub.columns]
    st.dataframe(sub[cols], use_container_width=True)

    hand_list = sub["hand_no"].dropna().astype(int).tolist()
    if hand_list:
        hand_no = st.selectbox("Select a hand number", hand_list, key="hand_pick")

        st.subheader(f"Full Hand Timeline: Hand #{hand_no}")
        hand_events = events[events["hand_no"] == hand_no].sort_values("idx").copy()

        show_cols = [c for c in ["idx", "street", "kind", "player", "position", "amount", "delta_put_in", "pot_before", "pot_after", "wetness", "raw"] if c in hand_events.columns]
        st.dataframe(hand_events[show_cols], use_container_width=True)
    else:
        st.caption("No hands available for this combo under the current filter.")
