
import argparse
import os
import pandas as pd
import numpy as np

from src.poker_parser import PokerLogParser


def make_action_summary(events: pd.DataFrame) -> pd.DataFrame:
    """
    Create action summary per (hand_no, player).
    """
    ev = events.copy()
    ev = ev[ev["player"].notna()].copy()
    ev["player"] = ev["player"].astype(str)

    def fmt_row(r):
        k = r["kind"]
        amt = r["amount"]
        d = r.get("delta_put_in", None)
        if k in ("bet", "raise", "call", "post_sb", "post_bb", "collected", "uncalled"):
            if pd.notna(amt):
                return f"{k} {int(amt)}"
            return k
        if k == "shows":
            sc = r.get("show_cards", None)
            return "shows"
        return k

    ev["tok"] = ev.apply(fmt_row, axis=1)

    # group by street to keep readable
    ev = ev.sort_values(["hand_no", "idx"])
    grouped = ev.groupby(["hand_no", "player", "street"])["tok"].apply(lambda x: ", ".join(x)).reset_index()

    # combine streets into one string
    def combine_streets(g):
        parts = []
        for st in ["PREFLOP", "FLOP", "TURN", "RIVER"]:
            sub = g[g["street"] == st]
            if len(sub) > 0:
                parts.append(f"{st}: {sub['tok'].iloc[0]}")
        return " | ".join(parts)

    out = grouped.groupby(["hand_no", "player"]).apply(combine_streets).reset_index(name="action_summary")
    return out


def build_hand_player_details(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build one row per (hand_no, player) with showdown/seen-street flags and shown combo.
    Uses only events_df so it works even without HandState objects.
    """
    ev = events_df.copy()
    ev["hand_no"] = ev["hand_no"].astype(int)

    # Identify street occurrences per hand
    streets_present = ev[ev["kind"] == "board"].groupby("hand_no")["street"].apply(set).to_dict()

    # Fold street per player per hand
    folds = ev[ev["kind"] == "fold"][["hand_no", "player", "street", "idx"]].copy()
    folds = folds.sort_values(["hand_no", "player", "idx"]).drop_duplicates(["hand_no", "player"])
    fold_street = {(int(r.hand_no), str(r.player)): str(r.street) for r in folds.itertuples(index=False)}

    # Showdown definition: river dealt + any shows in hand
    river_dealt = ev[(ev["kind"] == "board") & (ev["street"] == "RIVER")].groupby("hand_no").size()
    any_show = ev[ev["kind"] == "shows"].groupby("hand_no").size()
    showdown_hands = set(river_dealt.index).intersection(set(any_show.index))

    # Shown cards -> combo (you already computed combo in poker_parser.py; here we recompute minimally)
    # We'll read show_cards if it's present; if not, we can parse raw, but your events.parquet includes show_cards.
    shows = ev[ev["kind"] == "shows"][["hand_no", "player", "show_cards"]].copy()
    shows = shows.dropna(subset=["show_cards"])

    # collected amounts
    collected = ev[ev["kind"] == "collected"][["hand_no", "player", "amount"]].copy()
    collected = collected.groupby(["hand_no", "player"], as_index=False)["amount"].sum().rename(columns={"amount": "collected_amt"})

    # Who are players per hand?
    players_in_hand = ev[ev["player"].notna()][["hand_no", "player"]].drop_duplicates()

    # Seen street rule: player sees FLOP/TURN/RIVER if that street exists and they folded AFTER it (or never folded)
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
        row = {
            "hand_no": hn,
            "player": p,
            "saw_FLOP": sees(hn, p, "FLOP"),
            "saw_TURN": sees(hn, p, "TURN"),
            "saw_RIVER": sees(hn, p, "RIVER"),
            "showdown_happened": hn in showdown_hands,
            "folded": (hn, p) in fold_street,
        }
        rows.append(row)

    hp = pd.DataFrame(rows)

    # Attach shown combo (if any)
    # show_cards is a tuple like ('K♠','Q♣'); your poker_parser has _to_combo, but we’ll use it via a tiny inline parser:
    from src.poker_parser import parse_card, PokerLogParser  # reuse your existing parser helpers
    to_combo = PokerLogParser._to_combo

    def cards_to_combo(x):
        if isinstance(x, (list, tuple)) and len(x) == 2:
            return to_combo(x[0], x[1])
        return np.nan

    shows["combo"] = shows["show_cards"].apply(cards_to_combo)
    shows = shows.drop(columns=["show_cards"])

    hp = hp.merge(shows, on=["hand_no", "player"], how="left")
    hp = hp.merge(collected, on=["hand_no", "player"], how="left")
    hp["collected_amt"] = hp["collected_amt"].fillna(0).astype(int)
    hp["won_hand"] = hp["collected_amt"] > 0

    return hp


def build_ranges(hp: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Range rule (your current rule):
      - only count if player voluntarily continued preflop (called/raised/bet to >= BB)
      - AND showdown happened
      - AND combo is known (they showed)
    We'll reconstruct voluntary_preflop from PREFLOP call/bet/raise/allin amounts.
    """
    ev = events_df.copy()
    pre = ev[(ev["street"] == "PREFLOP") & (ev["kind"].isin(["call", "bet", "raise", "allin"]))].copy()
    # voluntary continuation = did a preflop action with amount >= 40 (BB)
    pre["amount"] = pd.to_numeric(pre["amount"], errors="coerce")
    vol = pre.groupby(["hand_no", "player"], as_index=False)["amount"].max()
    vol["voluntary_preflop"] = vol["amount"] >= 40
    vol = vol.drop(columns=["amount"])

    hp2 = hp.merge(vol, on=["hand_no", "player"], how="left")
    hp2["voluntary_preflop"] = hp2["voluntary_preflop"].fillna(False)

    eligible = hp2[(hp2["showdown_happened"]) & (hp2["voluntary_preflop"]) & (hp2["combo"].notna())].copy()

    ranges = eligible.groupby(["player", "combo"], as_index=False).size().rename(columns={"size": "count"})
    ranges["pct_of_observed_showdowns"] = ranges.groupby("player")["count"].transform(lambda x: x / x.sum())
    ranges = ranges.sort_values(["player", "count"], ascending=[True, False])
    return ranges, hp2




def build_sizing_tables(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build sizing distribution tables:
      - by player / street / wetness_bin / pot_bucket
    Uses bet/raise/allin events with pot_frac_bucket.
    """
    df = events_df.copy()

    df = df[df["kind"].isin(["bet", "raise", "allin"])].copy()
    df["player"] = df["player"].astype(str)

    # Wetness bins only for postflop
    def wet_bin(w):
        if pd.isna(w):
            return "PREFLOP"
        if w < 30:
            return "dry"
        if w < 60:
            return "medium"
        return "wet"

    df["wetness_bin"] = df["wetness"].apply(wet_bin)
    df["pot_bucket"] = df["pot_frac_bucket"].fillna("unknown")

    g = df.groupby(["player", "street", "wetness_bin", "pot_bucket"], dropna=False).size().reset_index(name="count")
    g["pct_within_context"] = g.groupby(["player", "street", "wetness_bin"])["count"].transform(lambda x: x / x.sum())
    return g.sort_values(["player", "street", "wetness_bin", "pct_within_context"], ascending=[True, True, True, False])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", type=str)
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--entry_col", type=str, default="entry")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    parser = PokerLogParser(bb_amt=40, sb_amt=20)
    events_df, summary_df = parser.parse_csv(args.csv_path, entry_col=args.entry_col)

    # --- Ensure required columns exist (until parser always provides them) ---
    if "n_players" not in events_df.columns:
        events_df["n_players"] = (
            events_df.groupby("hand_no")["player"]
            .transform(lambda s: s.dropna().nunique())
            .astype(int)
        )

    if "position" not in events_df.columns:
        events_df["position"] = np.nan

    # Ensure position is usable for grouping / display
    events_df["position"] = events_df["position"].fillna("UNK")

    # --- Core outputs ---
    events_path = os.path.join(args.outdir, "events.parquet")
    summary_path = os.path.join(args.outdir, "player_summary.csv")
    sizing_path = os.path.join(args.outdir, "sizing_by_wetness.csv")

    events_df.to_parquet(events_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    # --- New: hand_player_details + ranges ---
    action_summary = make_action_summary(events_df)
    hp = build_hand_player_details(events_df)

    # Bring in per-hand position + n_players from events
    pos = (
        events_df[events_df["player"].notna()][["hand_no", "player", "position", "n_players"]]
        .drop_duplicates()
    )
    hp = hp.merge(pos, on=["hand_no", "player"], how="left")
    hp = hp.merge(action_summary, on=["hand_no", "player"], how="left")

    ranges_df, hp2 = build_ranges(hp, events_df)

    hp_path = os.path.join(args.outdir, "hand_player_details.csv")
    ranges_path = os.path.join(args.outdir, "ranges.csv")
    ranges_by_pos_path = os.path.join(args.outdir, "ranges_by_position.csv")

    hp2.to_csv(hp_path, index=False)
    ranges_df.to_csv(ranges_path, index=False)

    # --- player × combo × position ---
    if "position" not in hp2.columns:
        hp2["position"] = "UNK"
    hp2["position"] = hp2["position"].fillna("UNK")

    ranges_by_pos = (
        hp2[(hp2["combo"].notna()) & (hp2["showdown_happened"]) & (hp2["voluntary_preflop"])]
        .groupby(["player", "combo", "position"], as_index=False, dropna=False)
        .size()
        .rename(columns={"size": "count"})
    )
    ranges_by_pos.to_csv(ranges_by_pos_path, index=False)

    print("DEBUG: ranges_by_pos rows =", len(ranges_by_pos))
    print("DEBUG: wrote", ranges_by_pos_path, "exists?", os.path.exists(ranges_by_pos_path))

    # --- Sizing ---
    sizing_df = build_sizing_tables(events_df)
    sizing_df.to_csv(sizing_path, index=False)

    print("Wrote:")
    print(" -", events_path)
    print(" -", summary_path)
    print(" -", sizing_path)
    print(" -", hp_path)
    print(" -", ranges_path)
    print(" -", ranges_by_pos_path)


if __name__ == "__main__":
    main()
