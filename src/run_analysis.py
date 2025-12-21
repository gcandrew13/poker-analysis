
import argparse
import os
import pandas as pd
import numpy as np

from src.poker_parser import PokerLogParser




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
    ap.add_argument("--outdir", type=str, default="poker_out")
    ap.add_argument("--entry_col", type=str, default="entry")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    parser = PokerLogParser(bb_amt=40, sb_amt=20)
    events_df, summary_df = parser.parse_csv(args.csv_path, entry_col=args.entry_col)

    # Save core outputs
    events_path = os.path.join(args.outdir, "events.parquet")
    summary_path = os.path.join(args.outdir, "player_summary.csv")

    events_df.to_parquet(events_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    # Sizing distribution tables
    sizing_df = build_sizing_tables(events_df)
    sizing_path = os.path.join(args.outdir, "sizing_by_wetness.csv")
    sizing_df.to_csv(sizing_path, index=False)

    print("Wrote:")
    print(" -", events_path)
    print(" -", summary_path)
    print(" -", sizing_path)


if __name__ == "__main__":
    main()
