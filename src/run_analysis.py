import argparse
import json
import os
from typing import Tuple

import numpy as np
import pandas as pd

from src.poker_parser import PokerLogParser


# -------------------------
# Blind detection
# -------------------------
def detect_blinds_from_csv(csv_path: str, entry_col: str) -> Tuple[float | None, float | None]:
    raw = pd.read_csv(csv_path)
    if entry_col not in raw.columns:
        raise ValueError(f"entry_col='{entry_col}' not found in CSV columns: {raw.columns.tolist()}")

    s = raw[entry_col].astype(str)

    sb_vals = s.str.extract(r"posts a small blind of\s+([0-9]+(?:\.[0-9]+)?)", expand=False)
    bb_vals = s.str.extract(r"posts a big blind of\s+([0-9]+(?:\.[0-9]+)?)", expand=False)

    sb_vals = pd.to_numeric(sb_vals, errors="coerce").dropna()
    bb_vals = pd.to_numeric(bb_vals, errors="coerce").dropna()

    sb = float(sb_vals.mode().iloc[0]) if len(sb_vals) else None
    bb = float(bb_vals.mode().iloc[0]) if len(bb_vals) else None
    return sb, bb


# -------------------------
# Action summary
# -------------------------
def make_action_summary(events: pd.DataFrame) -> pd.DataFrame:
    ev = events.copy()
    ev = ev[ev["player"].notna()].copy()
    ev["player"] = ev["player"].astype(str)

    def fmt_row(r):
        k = r["kind"]
        amt = r["amount"]
        if k in ("bet", "raise", "call", "post_sb", "post_bb", "collected", "uncalled", "allin"):
            if pd.notna(amt):
                # keep decimals if needed (0.5/1 games)
                try:
                    x = float(amt)
                    if abs(x - round(x)) < 1e-9:
                        return f"{k} {int(round(x))}"
                    return f"{k} {x:.2f}"
                except Exception:
                    return f"{k} {amt}"
            return k
        if k == "shows":
            return "shows"
        if k == "hole":
            return "hole"
        return k

    ev["tok"] = ev.apply(fmt_row, axis=1)

    ev = ev.sort_values(["hand_no", "idx"])
    grouped = (
        ev.groupby(["hand_no", "player", "street"])["tok"]
        .apply(lambda x: ", ".join(x))
        .reset_index()
    )

    def combine_streets(g):
        parts = []
        for st in ["PREFLOP", "FLOP", "TURN", "RIVER"]:
            sub = g[g["street"] == st]
            if len(sub) > 0:
                parts.append(f"{st}: {sub['tok'].iloc[0]}")
        return " | ".join(parts)

    out = grouped.groupby(["hand_no", "player"]).apply(combine_streets).reset_index(name="action_summary")
    return out


# -------------------------
# Hand-player details
# -------------------------
def build_hand_player_details(events_df: pd.DataFrame) -> pd.DataFrame:
    ev = events_df.copy()
    ev["hand_no"] = ev["hand_no"].astype(int)

    streets_present = ev[ev["kind"] == "board"].groupby("hand_no")["street"].apply(set).to_dict()

    folds = ev[ev["kind"] == "fold"][["hand_no", "player", "street", "idx"]].copy()
    folds = folds.sort_values(["hand_no", "player", "idx"]).drop_duplicates(["hand_no", "player"])
    fold_street = {(int(r.hand_no), str(r.player)): str(r.street) for r in folds.itertuples(index=False)}

    river_dealt = ev[(ev["kind"] == "board") & (ev["street"] == "RIVER")].groupby("hand_no").size()
    any_show = ev[ev["kind"] == "shows"].groupby("hand_no").size()
    showdown_hands = set(river_dealt.index).intersection(set(any_show.index))

    # shows -> combo
    shows = ev[ev["kind"] == "shows"][["hand_no", "player", "show_cards"]].copy()
    shows = shows.dropna(subset=["show_cards"])

    # hole -> combo (your hand)
    holes = ev[ev["kind"] == "hole"][["hand_no", "player", "show_cards"]].copy()
    holes = holes.dropna(subset=["show_cards"])

    collected = ev[ev["kind"] == "collected"][["hand_no", "player", "amount"]].copy()
    collected["amount"] = pd.to_numeric(collected["amount"], errors="coerce")
    collected = (
        collected.groupby(["hand_no", "player"], as_index=False)["amount"]
        .sum()
        .rename(columns={"amount": "collected_amt"})
    )

    players_in_hand = ev[ev["player"].notna()][["hand_no", "player"]].drop_duplicates()

    def sees(hand_no: int, player: str, street: str) -> bool:
        stset = streets_present.get(hand_no, set())
        if street not in stset:
            return False
        f = fold_street.get((hand_no, player))
        if f is None:
            return True
        order = {"PREFLOP": 0, "FLOP": 1, "TURN": 2, "RIVER": 3}
        return order[street] < order[f]

    rows = []
    for hn, p in players_in_hand.itertuples(index=False):
        hn = int(hn)
        p = str(p)
        rows.append(
            {
                "hand_no": hn,
                "player": p,
                "saw_FLOP": sees(hn, p, "FLOP"),
                "saw_TURN": sees(hn, p, "TURN"),
                "saw_RIVER": sees(hn, p, "RIVER"),
                "showdown_happened": hn in showdown_hands,
                "folded": (hn, p) in fold_street,
            }
        )

    hp = pd.DataFrame(rows)

    from src.poker_parser import PokerLogParser
    to_combo = PokerLogParser._to_combo

    def cards_to_combo(x):
        if isinstance(x, (list, tuple)) and len(x) == 2:
            c = to_combo(x[0], x[1])
            return c if c else np.nan
        return np.nan

    if not shows.empty:
        shows["combo_show"] = shows["show_cards"].apply(cards_to_combo)
        shows = shows.drop(columns=["show_cards"])
    else:
        shows = pd.DataFrame(columns=["hand_no", "player", "combo_show"])

    if not holes.empty:
        holes["combo_hole"] = holes["show_cards"].apply(cards_to_combo)
        holes = holes.drop(columns=["show_cards"])
    else:
        holes = pd.DataFrame(columns=["hand_no", "player", "combo_hole"])

    hp = hp.merge(shows, on=["hand_no", "player"], how="left")
    hp = hp.merge(holes, on=["hand_no", "player"], how="left")
    hp = hp.merge(collected, on=["hand_no", "player"], how="left")

    hp["collected_amt"] = hp["collected_amt"].fillna(0.0)
    hp["won_hand"] = hp["collected_amt"] > 0

    # unified combo: prefer show combo, else hole combo
    hp["combo"] = hp["combo_show"].combine_first(hp["combo_hole"])
    hp["hole_known_in_hand"] = hp["combo_hole"].notna()

    return hp


# -------------------------
# Ranges
# -------------------------
def build_ranges(hp: pd.DataFrame, events_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Produces:
      - ranges_showdown: combos from shows only
      - ranges_known_hole: combos from 'Your hand' (hole cards) and/or shows (unified via hp['combo'])
      - hp2: hp with voluntary_preflop flag
    """
    ev = events_df.copy()

    pre = ev[(ev["street"] == "PREFLOP") & (ev["kind"].isin(["call", "bet", "raise", "allin"]))].copy()
    pre["amount"] = pd.to_numeric(pre["amount"], errors="coerce")

    # VPIP/voluntary preflop: any call/bet/raise/allin amount > 0 (do NOT threshold by BB; SB completes count)
    pre["vpip_like"] = pre["amount"].fillna(0) > 0
    vol = pre.groupby(["hand_no", "player"], as_index=False)["vpip_like"].max().rename(columns={"vpip_like": "voluntary_preflop"})

    hp2 = hp.merge(vol, on=["hand_no", "player"], how="left")
    hp2["voluntary_preflop"] = hp2["voluntary_preflop"].fillna(False)

    # showdown-only range (must have combo_show)
    show_eligible = hp2[(hp2["showdown_happened"]) & (hp2["voluntary_preflop"]) & (hp2["combo_show"].notna())].copy()
    ranges_showdown = (
        show_eligible.groupby(["player", "combo_show"], as_index=False)
        .size()
        .rename(columns={"combo_show": "combo", "size": "count"})
    )
    if not ranges_showdown.empty:
        ranges_showdown["pct"] = ranges_showdown.groupby("player")["count"].transform(lambda x: x / x.sum())
        ranges_showdown = ranges_showdown.sort_values(["player", "count"], ascending=[True, False])

    # known-hole range (prefer show, else hole): must have unified combo and voluntary_preflop
    hole_eligible = hp2[(hp2["voluntary_preflop"]) & (hp2["combo"].notna())].copy()
    ranges_known_hole = (
        hole_eligible.groupby(["player", "combo"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    if not ranges_known_hole.empty:
        ranges_known_hole["pct"] = ranges_known_hole.groupby("player")["count"].transform(lambda x: x / x.sum())
        ranges_known_hole = ranges_known_hole.sort_values(["player", "count"], ascending=[True, False])

    return ranges_showdown, ranges_known_hole, hp2


# -------------------------
# Summary stats
# -------------------------
def compute_vpip(events_df: pd.DataFrame) -> pd.DataFrame:
    ev = events_df.copy()
    ev = ev[(ev["street"] == "PREFLOP") & (ev["player"].notna())].copy()

    dealt = ev[["hand_no", "player"]].drop_duplicates()
    denom = dealt.groupby("player")["hand_no"].nunique().rename("hands_dealt")

    vol = ev[ev["kind"].isin(["call", "raise", "bet", "allin"])][["hand_no", "player"]].drop_duplicates()
    numer = vol.groupby("player")["hand_no"].nunique().rename("vpip_hands")

    out = pd.concat([denom, numer], axis=1).fillna(0)
    out["vpip"] = out["vpip_hands"] / out["hands_dealt"].replace({0: np.nan})
    out = out.reset_index()
    return out


def compute_aggression(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggression freq by street = (bet+raise+allin events) / (all action events) per street.
    Action events = check/call/bet/raise/fold/allin.
    """
    ev = events_df.copy()
    ev = ev[ev["player"].notna()].copy()

    action_kinds = set(["check", "call", "bet", "raise", "fold", "allin"])
    aggr_kinds = set(["bet", "raise", "allin"])

    ev = ev[ev["kind"].isin(action_kinds)].copy()

    # opportunities = number of action events by player/street
    opp = ev.groupby(["player", "street"]).size().rename("opp").reset_index()

    ag = ev[ev["kind"].isin(aggr_kinds)].groupby(["player", "street"]).size().rename("aggr").reset_index()

    m = opp.merge(ag, on=["player", "street"], how="left").fillna({"aggr": 0})
    m["aggr_freq"] = m["aggr"] / m["opp"].replace({0: np.nan})

    # pivot to columns
    pivot = m.pivot(index="player", columns="street", values="aggr_freq").reset_index()
    for st in ["PREFLOP", "FLOP", "TURN", "RIVER"]:
        if st not in pivot.columns:
            pivot[st] = np.nan
    pivot = pivot.rename(columns={
        "PREFLOP": "aggr_freq_PREFLOP",
        "FLOP": "aggr_freq_FLOP",
        "TURN": "aggr_freq_TURN",
        "RIVER": "aggr_freq_RIVER",
    })
    return pivot


def write_debug(outdir: str, events_df: pd.DataFrame, hp2: pd.DataFrame, ranges_showdown: pd.DataFrame, ranges_hole: pd.DataFrame):
    os.makedirs(outdir, exist_ok=True)
    counts = {
        "events_rows": int(len(events_df)),
        "hands": int(events_df["hand_no"].nunique()) if "hand_no" in events_df.columns and len(events_df) else 0,
        "players": int(events_df["player"].dropna().nunique()) if "player" in events_df.columns and len(events_df) else 0,
        "kinds": events_df["kind"].value_counts().to_dict() if "kind" in events_df.columns and len(events_df) else {},
        "hp2_rows": int(len(hp2)),
        "hp2_combo_nonnull": int(hp2["combo"].notna().sum()) if "combo" in hp2.columns else 0,
        "hp2_combo_show_nonnull": int(hp2["combo_show"].notna().sum()) if "combo_show" in hp2.columns else 0,
        "hp2_combo_hole_nonnull": int(hp2["combo_hole"].notna().sum()) if "combo_hole" in hp2.columns else 0,
        "ranges_showdown_rows": int(len(ranges_showdown)),
        "ranges_known_hole_rows": int(len(ranges_hole)),
    }
    with open(os.path.join(outdir, "debug_counts.json"), "w") as f:
        json.dump(counts, f, indent=2)

    events_df.head(500).to_csv(os.path.join(outdir, "debug_events_sample.csv"), index=False)
    hp2.head(500).to_csv(os.path.join(outdir, "debug_hp2_sample.csv"), index=False)

    known_cards = events_df[events_df["kind"].isin(["hole", "shows"])].copy()
    known_cards.to_csv(os.path.join(outdir, "debug_known_cards_events.csv"), index=False)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", type=str)
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--entry_col", type=str, default="entry")

    ap.add_argument("--sb", type=float, default=None)
    ap.add_argument("--bb", type=float, default=None)
    ap.add_argument("--autodetect_blinds", action="store_true")

    ap.add_argument("--debug", action="store_true", help="Write debug_* artifacts to outdir")
    ap.add_argument("--verbose_hands", action="store_true", help="Print verbose per-hand parse summaries")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    sb = args.sb
    bb = args.bb
    if args.autodetect_blinds or sb is None or bb is None:
        dsb, dbb = detect_blinds_from_csv(args.csv_path, args.entry_col)
        if sb is None:
            sb = dsb
        if bb is None:
            bb = dbb

    if sb is None or bb is None:
        sb = 20.0 if sb is None else sb
        bb = 40.0 if bb is None else bb
        print(f"WARNING: Could not detect blinds; defaulting SB={sb}, BB={bb}")

    print(f"Using blinds: SB={sb}, BB={bb}")

    parser = PokerLogParser(bb_amt=float(bb), sb_amt=float(sb), verbose_hands=bool(args.verbose_hands))
    events_df, _ = parser.parse_csv(args.csv_path, entry_col=args.entry_col)

    # -------------------------
    # CHECKPOINT A: parse health
    # -------------------------
    print("\n=== CHECKPOINT A: after parse_csv ===")
    print("events rows:", len(events_df))
    if len(events_df) > 0:
        print("unique hands:", events_df["hand_no"].nunique())
        print("unique players:", events_df["player"].dropna().nunique())
        print("kind counts (top):")
        print(events_df["kind"].value_counts().head(20).to_string())

    # add n_players already computed in parser, but ensure exists
    if "n_players" not in events_df.columns:
        events_df["n_players"] = (
            events_df.groupby("hand_no")["player"].transform(lambda s: s.dropna().nunique()).astype(int)
        )
    if "position" not in events_df.columns:
        events_df["position"] = "UNK"
    events_df["position"] = events_df["position"].fillna("UNK")

    # hand details + action summary
    action_summary = make_action_summary(events_df)
    hp = build_hand_player_details(events_df)

    pos = (
        events_df[events_df["player"].notna()][["hand_no", "player", "position", "n_players"]]
        .drop_duplicates()
    )
    hp = hp.merge(pos, on=["hand_no", "player"], how="left")
    hp = hp.merge(action_summary, on=["hand_no", "player"], how="left")

    ranges_showdown, ranges_hole, hp2 = build_ranges(hp, events_df)

    # -------------------------
    # CHECKPOINT B: where combos come from
    # -------------------------
    print("\n=== CHECKPOINT B: hp2/ranges health ===")
    print("hp2 rows:", len(hp2))
    print("hp2 unique hands:", hp2["hand_no"].nunique() if len(hp2) else 0)
    print("hp2 combo_show non-null:", int(hp2["combo_show"].notna().sum()) if "combo_show" in hp2.columns else 0)
    print("hp2 combo_hole non-null:", int(hp2["combo_hole"].notna().sum()) if "combo_hole" in hp2.columns else 0)
    print("hp2 unified combo non-null:", int(hp2["combo"].notna().sum()) if "combo" in hp2.columns else 0)
    print("voluntary_preflop True rows:", int(hp2["voluntary_preflop"].sum()) if "voluntary_preflop" in hp2.columns else 0)
    print("ranges_showdown rows:", len(ranges_showdown))
    print("ranges_known_hole rows:", len(ranges_hole))

    # per-hand diagnostic (first 40 hands)
    print("\n=== CHECKPOINT C: per-hand diagnostic table (first 40) ===")
    hand_diag = (
        hp2.groupby("hand_no")
        .agg(
            n_players=("player", "nunique"),
            combos_any=("combo", lambda s: int(s.notna().sum())),
            combos_show=("combo_show", lambda s: int(s.notna().sum())),
            combos_hole=("combo_hole", lambda s: int(s.notna().sum())),
            any_showdown=("showdown_happened", "max"),
            any_vol=("voluntary_preflop", "max"),
        )
        .reset_index()
        .sort_values("hand_no")
    )
    print(hand_diag.head(40).to_string(index=False))

    # player summary computed from events
    hands_played = (
        events_df[events_df["player"].notna()][["hand_no", "player"]]
        .drop_duplicates()
        .groupby("player")["hand_no"]
        .nunique()
        .rename("hands_played")
        .reset_index()
    )
    vpip_df = compute_vpip(events_df)
    aggr_df = compute_aggression(events_df)

    player_summary = hands_played.merge(vpip_df, on="player", how="left").merge(aggr_df, on="player", how="left")

    # outputs
    events_path = os.path.join(args.outdir, "events.parquet")
    hp_path = os.path.join(args.outdir, "hand_player_details.csv")

    # IMPORTANT:
    # - ranges.csv will be the LARGE "known hole" range so the app shows your 100+ hands
    ranges_path = os.path.join(args.outdir, "ranges.csv")
    ranges_showdown_path = os.path.join(args.outdir, "ranges_showdown.csv")
    ranges_by_pos_path = os.path.join(args.outdir, "ranges_by_position.csv")

    summary_path = os.path.join(args.outdir, "player_summary.csv")

    events_df.to_parquet(events_path, index=False)
    hp2.to_csv(hp_path, index=False)
    player_summary.to_csv(summary_path, index=False)

    ranges_hole.to_csv(ranges_path, index=False)
    ranges_showdown.to_csv(ranges_showdown_path, index=False)

    # ranges by position (using unified combo)
    if "position" not in hp2.columns:
        hp2["position"] = "UNK"
    hp2["position"] = hp2["position"].fillna("UNK")

    ranges_by_pos = (
        hp2[(hp2["combo"].notna()) & (hp2["voluntary_preflop"])]
        .groupby(["player", "combo", "position"], as_index=False, dropna=False)
        .size()
        .rename(columns={"size": "count"})
    )
    ranges_by_pos.to_csv(ranges_by_pos_path, index=False)

    if args.debug:
        write_debug(args.outdir, events_df, hp2, ranges_showdown, ranges_hole)
        print("DEBUG wrote:", os.path.join(args.outdir, "debug_counts.json"))

    print("\nWrote:")
    print(" -", events_path)
    print(" -", summary_path)
    print(" -", hp_path)
    print(" -", ranges_path, "(known-hole default)")
    print(" -", ranges_showdown_path)
    print(" -", ranges_by_pos_path)


if __name__ == "__main__":
    main()
