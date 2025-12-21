from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, Counter

import numpy as np
import pandas as pd


# ----------------------------
# Card parsing + wetness
# ----------------------------

RANK_TO_VAL = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
    "J": 11, "Q": 12, "K": 13, "A": 14
}
HIGH_RANKS = {"A", "K", "Q", "J", "10"}

SUIT_CHARS = {"♠", "♥", "♦", "♣"}

_card_token_re = re.compile(r"\s*(10|[2-9JQKA])([♠♥♦♣])\s*")


def parse_card(token: str) -> Tuple[str, str, int]:
    """
    Returns (rank_str, suit_char, rank_value).
    token examples: 'K♠', '10♣', 'A♦'
    """
    m = _card_token_re.fullmatch(token.strip())
    if not m:
        raise ValueError(f"Could not parse card token: {token!r}")
    r, s = m.group(1), m.group(2)
    return r, s, RANK_TO_VAL[r]


def parse_card_list(cards_str: str) -> List[str]:
    """
    Parses "5♥, A♥, 2♥" or "[5♥, A♥, 2♥]" into ['5♥','A♥','2♥'] (preserving '10♠' tokens).
    """
    s = cards_str.strip()
    s = s.strip("[]")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    # normalize spacing
    return parts


def wetness_score(board_cards: List[str]) -> float:
    """
    Heuristic wetness score 0-100 based on:
      - high cards
      - suits (flush potential)
      - connectivity (straight potential)
      - pairedness (more static)
      - dynamic-ness (draw heavy boards are dynamic)
    Uses only current street board_cards.
    """
    if len(board_cards) < 3:
        return 0.0

    ranks = []
    suits = []
    for c in board_cards:
        r, s, v = parse_card(c)
        ranks.append((r, v))
        suits.append(s)

    rank_strs = [rv[0] for rv in ranks]
    rank_vals = sorted([rv[1] for rv in ranks])

    # High card component
    high_cnt = sum(1 for r in rank_strs if r in HIGH_RANKS)
    high_comp = min(30.0, 8.0 * high_cnt + (5.0 if max(rank_vals) >= 13 else 0.0))  # K/A boosts

    # Suit / flush component
    suit_counts = Counter(suits)
    max_suit = max(suit_counts.values())
    if len(board_cards) == 3:
        # flop
        if max_suit == 3:
            suit_comp = 40.0
        elif max_suit == 2:
            suit_comp = 25.0
        else:
            suit_comp = 5.0
    else:
        # turn/river: draw strength based on max suit count
        if max_suit >= 4:
            suit_comp = 40.0
        elif max_suit == 3:
            suit_comp = 25.0
        else:
            suit_comp = 5.0

    # Connectivity component (straighty-ness)
    # Use span + gaps heuristic
    unique_vals = sorted(set(rank_vals))
    span = unique_vals[-1] - unique_vals[0]
    # Count adjacent gaps
    gaps = sum(1 for i in range(1, len(unique_vals)) if unique_vals[i] - unique_vals[i-1] >= 2)

    if span <= 4:
        conn_comp = 35.0
    elif span <= 6:
        conn_comp = 22.0
    elif span <= 8:
        conn_comp = 12.0
    else:
        conn_comp = 5.0

    # Penalize disconnected boards
    conn_comp -= 4.0 * gaps
    conn_comp = max(0.0, conn_comp)

    # Pairedness (more static -> reduce)
    counts = Counter(rank_vals)
    paired = any(v >= 2 for v in counts.values())
    trips = any(v >= 3 for v in counts.values())
    pair_pen = 10.0 if paired else 0.0
    pair_pen += 8.0 if trips else 0.0

    # Dynamic bonus: draws present
    flush_draw_present = (max_suit >= 2 and len(board_cards) == 3) or (max_suit >= 3 and len(board_cards) == 4)
    straight_draw_present = (span <= 6)  # coarse
    dyn_bonus = 10.0 if (flush_draw_present or straight_draw_present) else 0.0

    score = high_comp + suit_comp + conn_comp + dyn_bonus - pair_pen
    return float(np.clip(score, 0.0, 100.0))


def bucket_pot_fraction(x: Optional[float]) -> str:
    if x is None or np.isnan(x):
        return "unknown"
    if x < 0.25:
        return "<1/4"
    if x < 0.45:
        return "~1/3"
    if x < 0.60:
        return "~1/2"
    if x < 0.85:
        return "~3/4"
    if x < 1.25:
        return "~pot"
    return "overbet"


# ----------------------------
# Event structures
# ----------------------------

STREETS = ["PREFLOP", "FLOP", "TURN", "RIVER"]


@dataclass
class ActionEvent:
    hand_no: int
    street: str
    idx: int  # global row index in chronological stream
    raw: str

    player: Optional[str] = None
    kind: str = ""  # e.g. 'call','bet','raise','fold','check','post_sb','post_bb','uncalled','collected','shows','mucks','board'
    amount: Optional[int] = None  # amount in chips for the action (interpretation depends on kind)
    pot_before: Optional[int] = None
    pot_after: Optional[int] = None
    delta_put_in: Optional[int] = None  # for bet/call/raise: how many chips added now
    pot_frac: Optional[float] = None  # delta_put_in / pot_before for bet/raise/bet-like events
    pot_frac_bucket: Optional[str] = None

    board: List[str] = field(default_factory=list)
    wetness: Optional[float] = None

    # For showdown parsing
    show_cards: Optional[Tuple[str, str]] = None


@dataclass
class HandState:
    hand_no: int
    hand_id: Optional[str] = None
    dealer: Optional[str] = None  # may appear in log; not required
    players: List[str] = field(default_factory=list)
    seat_map: Dict[str, int] = field(default_factory=dict)

    sb_player: Optional[str] = None
    bb_player: Optional[str] = None
    sb_amt: int = 20
    bb_amt: int = 40

    board: List[str] = field(default_factory=list)
    street: str = "PREFLOP"
    pot: int = 0
    contrib_street: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    active: Dict[str, bool] = field(default_factory=dict)

    # For VPIP / "voluntarily made it past preflop"
    preflop_total_put_in: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    vpip_flag: Dict[str, bool] = field(default_factory=lambda: defaultdict(bool))
    voluntary_preflop_flag: Dict[str, bool] = field(default_factory=lambda: defaultdict(bool))

    # Positions inferred
    position: Dict[str, str] = field(default_factory=dict)

    # Showdown/winners
    river_dealt: bool = False
    any_show: bool = False
    collected: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    shown_cards: Dict[str, Tuple[str, str]] = field(default_factory=dict)

    # For opportunities/aggression counts
    opp: Dict[Tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))   # (player, street) -> opp count
    aggr: Dict[Tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))  # (player, street) -> bet/raise count

    # Sizing details
    sizing_rows: List[Dict[str, Any]] = field(default_factory=list)


# ----------------------------
# Regex patterns
# ----------------------------

RE_ENDING = re.compile(r"^-- ending hand #(?P<handno>\d+)\s*--$")
RE_STARTING = re.compile(
    r"^-- starting hand #(?P<handno>\d+)\s*\(id:\s*(?P<handid>[^)]+)\)\s*\(No Limit Texas Hold'em\)\s*"
    r"(?:\(dealer:\s*\"(?P<dealer>[^\"]+)\"\))?\s*--$"
)

RE_STACKS = re.compile(r"^Player stacks:\s*(?P<rest>.+)$")
RE_STACK_ITEM = re.compile(r"#(?P<seat>\d+)\s+\"(?P<player>[^\"]+)\"\s+\((?P<stack>\d+)\)")

RE_POST_BLIND = re.compile(r"^\"(?P<player>[^\"]+)\"\s+posts a\s+(?P<blind>small|big)\s+blind of\s+(?P<amt>\d+)$")

RE_FOLD_CHECK = re.compile(r"^\"(?P<player>[^\"]+)\"\s+(?P<verb>folds|checks)$")
RE_CALL = re.compile(r"^\"(?P<player>[^\"]+)\"\s+calls\s+(?P<amt>\d+)$")
RE_BET = re.compile(r"^\"(?P<player>[^\"]+)\"\s+bets\s+(?P<amt>\d+)$")
RE_RAISE = re.compile(r"^\"(?P<player>[^\"]+)\"\s+raises to\s+(?P<amt>\d+)(?:\s+and go all in)?$")

RE_ALLIN = re.compile(r"^\"(?P<player>[^\"]+)\"\s+go all in(?:\s+for\s+(?P<amt>\d+))?$")

RE_UNCALLED = re.compile(r"^Uncalled bet of\s+(?P<amt>\d+)\s+returned to\s+\"(?P<player>[^\"]+)\"$")
RE_COLLECTED = re.compile(r"^\"(?P<player>[^\"]+)\"\s+collected\s+(?P<amt>\d+)\s+from pot(?:\s+with\s+(?P<desc>.+))?$")

RE_SHOWS = re.compile(r"^\"(?P<player>[^\"]+)\"\s+shows a\s+(?P<c1>[^,]+),\s*(?P<c2>[^.]+)\.$")
RE_MUCKS = re.compile(r"^\"(?P<player>[^\"]+)\"\s+mucks$")

RE_FLOP = re.compile(r"^Flop:\s*\[(?P<cards>.+)\]\s*$")
RE_TURN = re.compile(r"^Turn:\s*(?P<pre>.+?)\s*\[(?P<card>.+)\]\s*$")
RE_RIVER = re.compile(r"^River:\s*(?P<pre>.+?)\s*\[(?P<card>.+)\]\s*$")


# ----------------------------
# Position inference
# ----------------------------

def position_labels(n_players: int) -> List[str]:
    """
    Returns labels in order of action preflop starting from UTG (first to act preflop),
    ending with SB, BB at the end (standard full ring convention).
    For short-handed, labels can overlap; we use composites.
    """
    if n_players <= 1:
        return ["UNK"]
    if n_players == 2:
        # Heads-up: dealer is SB; preflop first to act is SB (button)
        return ["BTN/SB", "BB"]
    if n_players == 3:
        # 3-handed: first to act preflop is BTN (also UTG)
        return ["BTN/UTG", "SB", "BB"]

    # 4+ players: UTG ... BTN, SB, BB
    # Build a "middle" sequence that fits n-3 positions between UTG and BTN inclusive.
    middle_count = n_players - 3  # positions excluding SB, BB, plus BTN included in middle
    # Common names from early to late position:
    base = ["UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO", "BTN"]
    # Ensure enough labels:
    if middle_count <= len(base):
        mids = base[:middle_count - 1] + ["BTN"] if middle_count >= 1 else ["BTN"]
    else:
        # Extend with UTG+K until we can end with BTN
        mids = []
        for k in range(middle_count - 1):
            mids.append(f"UTG+{k}" if k > 0 else "UTG")
        mids.append("BTN")

    return mids + ["SB", "BB"]


def infer_positions(
    players: List[str],
    sb_player: str,
    bb_player: str,
    preflop_actor_order: List[str],
) -> Dict[str, str]:
    """
    Infer positions based on blind posters and the rotation of preflop first actions.

    preflop_actor_order: list of players in the order they first appear in PREFLOP actions
      (fold/call/raise/bet/check) after blinds are posted.

    We assign:
      - SB and BB explicitly
      - the first actor in preflop_actor_order that's not SB/BB becomes first label (UTG or BTN/UTG, etc.)
      - continue for remaining players (excluding SB/BB) and force the last of the non-blinds to be BTN.

    This method matches your "button = last to act before action returns to SB" rule.
    """
    pos = {p: "UNK" for p in players}
    pos[sb_player] = "SB"
    pos[bb_player] = "BB"

    n = len(players)
    labels = position_labels(n)

    # Build circular order of non-blinds based on observed preflop first-actions.
    # Keep only seated players.
    seen = []
    for p in preflop_actor_order:
        if p in pos and p not in seen:
            seen.append(p)

    # Remove blinds from that sequence for assignment
    nonblind_seen = [p for p in seen if p not in (sb_player, bb_player)]
    nonblind_players = [p for p in players if p not in (sb_player, bb_player)]

    # If we didn't observe everyone (rare), append missing non-blinds at end as unknown order.
    for p in nonblind_players:
        if p not in nonblind_seen:
            nonblind_seen.append(p)

    # Labels structure:
    # labels = [<n-2 positions including BTN> , SB, BB]
    nonblind_labels = labels[:-2]  # includes BTN somewhere at end for n>=3
    # If lengths mismatch, trim/pad.
    k = min(len(nonblind_labels), len(nonblind_seen))
    for i in range(k):
        pos[nonblind_seen[i]] = nonblind_labels[i]

    # Override composites for short-handed correctness
    if n == 2:
        pos[sb_player] = "BTN/SB"
        pos[bb_player] = "BB"
    if n == 3:
        # If sb/bb assigned, first nonblind is BTN/UTG; but in 3-handed, dealer may be BTN and SB posted by next seat.
        # Keep "BTN/UTG" label for the first nonblind actor.
        pass

    return pos


# ----------------------------
# Main parser
# ----------------------------

class PokerLogParser:
    def __init__(self, bb_amt: int = 40, sb_amt: int = 20):
        self.bb_amt = bb_amt
        self.sb_amt = sb_amt

    def _apply_contribution(self, hand: HandState, player: str, new_total_this_street: int) -> Tuple[int, int]:
        """
        Update pot based on "calls X / bets X / raises to X" semantics:
          - X is total put in by that player THIS street
          - delta = X - current_contrib_this_street
        Returns (delta, pot_before)
        """
        pot_before = hand.pot
        prev = hand.contrib_street[player]
        delta = new_total_this_street - prev
        if delta < 0:
            # Data glitch or multi-action lines; treat as 0 and keep consistent
            delta = 0
        hand.contrib_street[player] = new_total_this_street
        hand.pot += delta
        return delta, pot_before

    def _new_street(self, hand: HandState, street: str, board_cards: List[str]):
        hand.street = street
        hand.contrib_street = defaultdict(int)  # reset per street contributions
        hand.board = board_cards

    def parse_csv(self, csv_path: str, entry_col: str = "entry") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns:
          - events_df: one row per parsed event (actions, boards, collected, etc.)
          - summary_df: per-player summary
        """
        df = pd.read_csv(csv_path)

        # Prefer 'order' if present; otherwise reverse the dataframe.
        if "order" in df.columns:
            df = df.sort_values("order", ascending=True).reset_index(drop=True)
        else:
            df = df.iloc[::-1].reset_index(drop=True)

        lines = df[entry_col].astype(str).tolist()

        hands: List[HandState] = []
        events: List[ActionEvent] = []

        current: Optional[HandState] = None
        global_idx = 0

        # We reconstruct hands by scanning chronological lines:
        # starting hand -> actions -> ending hand.
        for raw in lines:
            raw = raw.strip()

            m_end = RE_ENDING.match(raw)
            if m_end:
                # End previous hand (the one that just completed)
                # Note: In your log, "-- ending hand #X --" appears at the TOP of the finished hand block.
                # Chronologically, it will appear after the hand actions. We'll close on seeing it.
                if current is not None:
                    hands.append(current)
                    current = None
                global_idx += 1
                continue

            m_start = RE_STARTING.match(raw)
            if m_start:
                handno = int(m_start.group("handno"))
                handid = (m_start.group("handid") or "").strip()
                dealer = m_start.group("dealer")
                current = HandState(
                    hand_no=handno,
                    hand_id=handid,
                    dealer=dealer,
                    sb_amt=self.sb_amt,
                    bb_amt=self.bb_amt,
                )
                global_idx += 1
                continue

            if current is None:
                # Ignore lines outside a hand
                global_idx += 1
                continue

            # Player stacks
            m_st = RE_STACKS.match(raw)
            if m_st:
                rest = m_st.group("rest")
                players = []
                seat_map = {}
                for it in RE_STACK_ITEM.finditer(rest):
                    seat = int(it.group("seat"))
                    player = it.group("player")
                    players.append(player)
                    seat_map[player] = seat
                current.players = players
                current.seat_map = seat_map
                current.active = {p: True for p in players}
                global_idx += 1
                continue

            # Blinds
            m_bl = RE_POST_BLIND.match(raw)
            if m_bl:
                p = m_bl.group("player")
                blind = m_bl.group("blind")
                amt = int(m_bl.group("amt"))
                if blind == "small":
                    current.sb_player = p
                    current.sb_amt = amt
                    current.pot += amt
                    current.preflop_total_put_in[p] += amt
                else:
                    current.bb_player = p
                    current.bb_amt = amt
                    current.pot += amt
                    current.preflop_total_put_in[p] += amt

                ev = ActionEvent(
                    hand_no=current.hand_no,
                    street="PREFLOP",
                    idx=global_idx,
                    raw=raw,
                    player=p,
                    kind="post_sb" if blind == "small" else "post_bb",
                    amount=amt,
                    pot_before=current.pot - amt,
                    pot_after=current.pot,
                )
                events.append(ev)
                global_idx += 1
                continue

            # Board streets
            m_fl = RE_FLOP.match(raw)
            if m_fl:
                cards = parse_card_list(m_fl.group("cards"))
                self._new_street(current, "FLOP", cards)
                w = wetness_score(cards)
                ev = ActionEvent(current.hand_no, "FLOP", global_idx, raw, kind="board", board=cards, wetness=w)
                events.append(ev)
                global_idx += 1
                continue

            m_tu = RE_TURN.match(raw)
            if m_tu:
                # board shown as "a, b, c [turn]"
                pre = parse_card_list(m_tu.group("pre"))
                card = m_tu.group("card").strip()
                cards = pre + [card]
                self._new_street(current, "TURN", cards)
                w = wetness_score(cards)
                ev = ActionEvent(current.hand_no, "TURN", global_idx, raw, kind="board", board=cards, wetness=w)
                events.append(ev)
                global_idx += 1
                continue

            m_ri = RE_RIVER.match(raw)
            if m_ri:
                pre = parse_card_list(m_ri.group("pre"))
                card = m_ri.group("card").strip()
                cards = pre + [card]
                self._new_street(current, "RIVER", cards)
                current.river_dealt = True
                w = wetness_score(cards)
                ev = ActionEvent(current.hand_no, "RIVER", global_idx, raw, kind="board", board=cards, wetness=w)
                events.append(ev)
                global_idx += 1
                continue

            # Uncalled
            m_unc = RE_UNCALLED.match(raw)
            if m_unc:
                p = m_unc.group("player")
                amt = int(m_unc.group("amt"))
                pot_before = current.pot
                current.pot = max(0, current.pot - amt)
                ev = ActionEvent(
                    current.hand_no, current.street, global_idx, raw,
                    player=p, kind="uncalled", amount=amt,
                    pot_before=pot_before, pot_after=current.pot
                )
                events.append(ev)
                global_idx += 1
                continue

            # Collected (winner)
            m_col = RE_COLLECTED.match(raw)
            if m_col:
                p = m_col.group("player")
                amt = int(m_col.group("amt"))
                current.collected[p] += amt
                ev = ActionEvent(
                    current.hand_no, current.street, global_idx, raw,
                    player=p, kind="collected", amount=amt
                )
                events.append(ev)
                global_idx += 1
                continue

            # Shows / mucks
            m_sh = RE_SHOWS.match(raw)
            if m_sh:
                p = m_sh.group("player")
                c1 = m_sh.group("c1").strip()
                c2 = m_sh.group("c2").strip()
                current.any_show = True
                current.shown_cards[p] = (c1, c2)
                ev = ActionEvent(
                    current.hand_no, current.street, global_idx, raw,
                    player=p, kind="shows", show_cards=(c1, c2)
                )
                events.append(ev)
                global_idx += 1
                continue

            m_mu = RE_MUCKS.match(raw)
            if m_mu:
                p = m_mu.group("player")
                ev = ActionEvent(current.hand_no, current.street, global_idx, raw, player=p, kind="mucks")
                events.append(ev)
                global_idx += 1
                continue

            # Calls / Bets / Raises / All-in / Fold / Check
            # Note: We count "opportunities" only when the player takes an action on that street.
            m_fc = RE_FOLD_CHECK.match(raw)
            if m_fc:
                p = m_fc.group("player")
                verb = m_fc.group("verb")
                # opportunities
                current.opp[(p, current.street)] += 1
                if verb == "folds":
                    current.active[p] = False
                    kind = "fold"
                else:
                    kind = "check"

                ev = ActionEvent(current.hand_no, current.street, global_idx, raw, player=p, kind=kind)
                events.append(ev)
                global_idx += 1
                continue

            m_ca = RE_CALL.match(raw)
            if m_ca:
                p = m_ca.group("player")
                amt = int(m_ca.group("amt"))

                current.opp[(p, current.street)] += 1  # had chance to raise too

                delta, pot_before = self._apply_contribution(current, p, amt)

                if current.street == "PREFLOP":
                    current.preflop_total_put_in[p] = max(current.preflop_total_put_in[p], amt)
                    # VPIP: voluntarily put money in preflop beyond forced (posting doesn't count)
                    # You asked to exclude BB posting; calls count.
                    current.vpip_flag[p] = True
                    # Voluntary preflop continuation (for range rule)
                    if amt >= current.bb_amt:
                        current.voluntary_preflop_flag[p] = True

                ev = ActionEvent(
                    current.hand_no, current.street, global_idx, raw,
                    player=p, kind="call", amount=amt,
                    pot_before=pot_before, pot_after=current.pot, delta_put_in=delta
                )
                events.append(ev)
                global_idx += 1
                continue

            m_be = RE_BET.match(raw)
            if m_be:
                p = m_be.group("player")
                amt = int(m_be.group("amt"))

                current.opp[(p, current.street)] += 1
                current.aggr[(p, current.street)] += 1

                delta, pot_before = self._apply_contribution(current, p, amt)
                pot_frac = (delta / pot_before) if pot_before > 0 else np.nan
                bucket = bucket_pot_fraction(pot_frac)

                # VPIP/voluntary continuation if preflop bet
                if current.street == "PREFLOP":
                    current.vpip_flag[p] = True
                    if amt >= current.bb_amt:
                        current.voluntary_preflop_flag[p] = True

                # record sizing
                current.sizing_rows.append({
                    "hand_no": current.hand_no,
                    "player": p,
                    "street": current.street,
                    "action": "bet",
                    "delta": delta,
                    "amount_total_street": amt,
                    "pot_before": pot_before,
                    "pot_frac": pot_frac,
                    "pot_bucket": bucket,
                    "wetness": wetness_score(current.board) if current.street != "PREFLOP" else np.nan,
                })

                ev = ActionEvent(
                    current.hand_no, current.street, global_idx, raw,
                    player=p, kind="bet", amount=amt,
                    pot_before=pot_before, pot_after=current.pot, delta_put_in=delta,
                    pot_frac=pot_frac, pot_frac_bucket=bucket,
                    wetness=wetness_score(current.board) if current.street != "PREFLOP" else None,
                    board=list(current.board)
                )
                events.append(ev)
                global_idx += 1
                continue

            m_ra = RE_RAISE.match(raw)
            if m_ra:
                p = m_ra.group("player")
                amt = int(m_ra.group("amt"))

                current.opp[(p, current.street)] += 1
                current.aggr[(p, current.street)] += 1

                delta, pot_before = self._apply_contribution(current, p, amt)
                pot_frac = (delta / pot_before) if pot_before > 0 else np.nan
                bucket = bucket_pot_fraction(pot_frac)

                if current.street == "PREFLOP":
                    current.vpip_flag[p] = True
                    if amt >= current.bb_amt:
                        current.voluntary_preflop_flag[p] = True

                current.sizing_rows.append({
                    "hand_no": current.hand_no,
                    "player": p,
                    "street": current.street,
                    "action": "raise",
                    "delta": delta,
                    "amount_total_street": amt,
                    "pot_before": pot_before,
                    "pot_frac": pot_frac,
                    "pot_bucket": bucket,
                    "wetness": wetness_score(current.board) if current.street != "PREFLOP" else np.nan,
                })

                ev = ActionEvent(
                    current.hand_no, current.street, global_idx, raw,
                    player=p, kind="raise", amount=amt,
                    pot_before=pot_before, pot_after=current.pot, delta_put_in=delta,
                    pot_frac=pot_frac, pot_frac_bucket=bucket,
                    wetness=wetness_score(current.board) if current.street != "PREFLOP" else None,
                    board=list(current.board)
                )
                events.append(ev)
                global_idx += 1
                continue

            m_ai = RE_ALLIN.match(raw)
            if m_ai:
                p = m_ai.group("player")
                amt_s = m_ai.group("amt")
                amt = int(amt_s) if amt_s else None

                current.opp[(p, current.street)] += 1
                current.aggr[(p, current.street)] += 1  # treat as aggressive

                pot_before = current.pot
                delta = None
                if amt is not None:
                    delta, pot_before2 = self._apply_contribution(current, p, amt)
                    pot_before = pot_before2
                    pot_frac = (delta / pot_before) if pot_before > 0 else np.nan
                    bucket = bucket_pot_fraction(pot_frac)
                else:
                    pot_frac = np.nan
                    bucket = "unknown"

                if current.street == "PREFLOP":
                    current.vpip_flag[p] = True
                    if amt is None or amt >= current.bb_amt:
                        current.voluntary_preflop_flag[p] = True

                ev = ActionEvent(
                    current.hand_no, current.street, global_idx, raw,
                    player=p, kind="allin", amount=amt,
                    pot_before=pot_before, pot_after=current.pot,
                    delta_put_in=delta, pot_frac=pot_frac, pot_frac_bucket=bucket,
                    wetness=wetness_score(current.board) if current.street != "PREFLOP" else None,
                    board=list(current.board)
                )
                events.append(ev)
                global_idx += 1
                continue

            # Unknown line inside a hand: keep as raw for debugging (optional)
            ev = ActionEvent(current.hand_no, current.street, global_idx, raw, kind="unknown")
            events.append(ev)
            global_idx += 1

        # If last hand wasn't closed by an ending line
        if current is not None:
            hands.append(current)

        # After parsing, infer positions for each hand using preflop action order
        events_df = pd.DataFrame([e.__dict__ for e in events])

        # Build per-hand preflop actor order (first appearance of each player in PREFLOP actions, excluding blind posts)
        for h in hands:
            # find hand events
            he = events_df[(events_df["hand_no"] == h.hand_no) & (events_df["street"] == "PREFLOP")]
            # order by idx (chronological)
            he = he.sort_values("idx")
            preflop_actor_order = []
            for _, r in he.iterrows():
                if r["kind"] in ("post_sb", "post_bb", "board", "unknown"):
                    continue
                p = r.get("player")
                if isinstance(p, str) and p and p not in preflop_actor_order:
                    preflop_actor_order.append(p)

            if h.sb_player and h.bb_player and h.players:
                h.position = infer_positions(h.players, h.sb_player, h.bb_player, preflop_actor_order)

        # Compute summary stats
        summary_df = self._compute_summary(hands, events_df)

        return events_df, summary_df

    def _compute_summary(self, hands: List[HandState], events_df: pd.DataFrame) -> pd.DataFrame:
        # Aggregators
        hands_dealt = Counter()
        vpip = Counter()
        saw = Counter()  # (player, street) saw street
        won_after = Counter()  # (player, street) win among those who saw that street
        showdown_cnt = Counter()
        aggr = Counter()  # (player, street)
        opp = Counter()   # (player, street)
        pos_cnt = Counter()  # (player, position)

        # Observed range
        observed_range = defaultdict(Counter)  # player -> Counter of combos like 'KQo','T8s'
        observed_range_hands = Counter()  # #hands contributing to range

        # Determine who "saw flop/turn/river" by fold timing:
        # We'll use events to see first board marker and whether player had folded before it.
        # Build per-hand fold street for each player.
        fold_street = defaultdict(dict)  # hand_no -> player -> street at which they folded (or None)
        for _, r in events_df.sort_values(["hand_no", "idx"]).iterrows():
            hn = int(r["hand_no"])
            k = r["kind"]
            p = r.get("player")
            st = r.get("street")
            if k == "fold" and isinstance(p, str):
                if p not in fold_street[hn]:
                    fold_street[hn][p] = st

        # Determine which streets occurred in each hand
        hand_has_street = defaultdict(set)
        for _, r in events_df.iterrows():
            if r["kind"] == "board":
                hand_has_street[int(r["hand_no"])].add(r["street"])

        # For each hand: tally
        for h in hands:
            for p in h.players:
                hands_dealt[p] += 1
                if h.vpip_flag.get(p, False):
                    # Exclude ONLY "posting BB" as a VPIP event:
                    # Our vpip_flag only gets set on call/bet/raise/allin preflop (not blind posting), so ok.
                    vpip[p] += 1

                # positions
                if h.position and p in h.position:
                    pos_cnt[(p, h.position[p])] += 1

                # saw streets if street exists AND player not folded before it
                for st in ["FLOP", "TURN", "RIVER"]:
                    if st in hand_has_street[h.hand_no]:
                        f = fold_street[h.hand_no].get(p)
                        # If they folded on PREFLOP they didn't see flop; folded on FLOP doesn't see turn, etc.
                        if f is None:
                            saw[(p, st)] += 1
                        else:
                            # Player folded at street f; they see only streets strictly before f
                            if STREETS.index(st) < STREETS.index(f):
                                saw[(p, st)] += 1

                # win flags: if collected > 0
                did_win = (h.collected.get(p, 0) > 0)
                if did_win:
                    for st in ["FLOP", "TURN", "RIVER"]:
                        if saw[(p, st)] > 0:
                            # We only want to count wins among hands where player saw st
                            # We'll add 1 per hand if saw street
                            f = fold_street[h.hand_no].get(p)
                            if f is None or STREETS.index(st) < STREETS.index(f):
                                won_after[(p, st)] += 1

                # showdown percent: your definition = river dealt + at least one shows
                showdown_happened = (h.river_dealt and h.any_show)
                if showdown_happened:
                    # player counted if they did NOT fold at any point
                    f = fold_street[h.hand_no].get(p)
                    if f is None:
                        showdown_cnt[p] += 1

            # Aggression/opportunities stored on HandState
            for (p, st), n in h.opp.items():
                opp[(p, st)] += n
            for (p, st), n in h.aggr.items():
                aggr[(p, st)] += n

            # Observed range: only if showdown happened AND player voluntarily continued preflop (your clarified rule)
            showdown_happened = (h.river_dealt and h.any_show)
            if showdown_happened:
                for p, (c1, c2) in h.shown_cards.items():
                    if not h.voluntary_preflop_flag.get(p, False):
                        continue
                    combo = self._to_combo(c1, c2)
                    observed_range[p][combo] += 1
                    observed_range_hands[p] += 1

        # Build summary df
        players = sorted(hands_dealt.keys())

        rows = []
        for p in players:
            hd = hands_dealt[p]
            vp = vpip[p]
            row = {
                "player": p,
                "hands_dealt": hd,
                "VPIP": vp / hd if hd else np.nan,
                "showdown_pct": showdown_cnt[p] / hd if hd else np.nan,
            }

            for st in ["PREFLOP", "FLOP", "TURN", "RIVER"]:
                a = aggr[(p, st)]
                o = opp[(p, st)]
                row[f"aggr_freq_{st}"] = a / o if o else np.nan
                row[f"opp_{st}"] = o
                row[f"aggr_{st}"] = a

            for st in ["FLOP", "TURN", "RIVER"]:
                s = saw[(p, st)]
                w = won_after[(p, st)]
                row[f"saw_{st}"] = s
                row[f"win_pct_if_saw_{st}"] = (w / s) if s else np.nan

            # Top positions
            # We'll add a few common position columns
            for pos in ["UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO", "BTN", "SB", "BB", "BTN/SB", "BTN/UTG"]:
                row[f"pos_{pos}"] = pos_cnt[(p, pos)]

            # Range summary
            if observed_range_hands[p] > 0:
                top = observed_range[p].most_common(5)
                row["range_top5"] = ", ".join([f"{k}:{v}" for k, v in top])
                row["range_hands_counted"] = observed_range_hands[p]
            else:
                row["range_top5"] = ""
                row["range_hands_counted"] = 0

            rows.append(row)

        return pd.DataFrame(rows)

    @staticmethod
    def _to_combo(card1: str, card2: str) -> str:
        """
        Convert shown cards to combo notation: e.g. 'K♠','Q♣' -> 'KQo'
        Rules:
          - higher rank first
          - suited => 's', offsuit => 'o'
        """
        r1, s1, v1 = parse_card(card1)
        r2, s2, v2 = parse_card(card2)

        # order by rank value desc; break ties by suit for stability
        if (v2 > v1) or (v2 == v1 and s2 > s1):
            r1, s1, v1, r2, s2, v2 = r2, s2, v2, r1, s1, v1

        suited = (s1 == s2)
        suffix = "s" if suited else "o"
        return f"{r1}{r2}{suffix}"
