import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd


# -------------------------
# Card helpers
# -------------------------
_RANK_ORDER = ["A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2"]
_RANK_INDEX = {r: i for i, r in enumerate(_RANK_ORDER)}

_CARD_RE = re.compile(r"(?P<rank>10|[2-9AJQK])(?P<suit>[♠♥♦♣])")

def parse_card(card_str: str) -> Optional[Tuple[str, str]]:
    """
    Returns (rank, suit) where rank is one of: A,K,Q,J,10,9..2 and suit is ♠♥♦♣.
    """
    if not isinstance(card_str, str):
        return None
    card_str = card_str.strip()
    m = _CARD_RE.search(card_str)
    if not m:
        return None
    rank = m.group("rank")
    suit = m.group("suit")
    return rank, suit

def normalize_rank(rank: str) -> str:
    if rank == "T":
        return "10"
    return rank

def cards_from_bracket_list(s: str) -> List[str]:
    """
    Extracts card strings like ['A♠','10♦'] from fragments like:
      "Flop:  [3♣, 2♥, 9♦]"
      "Turn: 3♣, 2♥, 9♦ [8♣]"
      "River: 3♣, 2♥, 9♦, 8♣ [A♠]"
    """
    if not isinstance(s, str):
        return []
    cards = _CARD_RE.findall(s)
    out = []
    for r, suit in cards:
        r = normalize_rank(r)
        out.append(f"{r}{suit}")
    return out


def pot_bucket(pot_frac: float) -> str:
    if pot_frac is None or pd.isna(pot_frac):
        return "unknown"
    if pot_frac <= 0.25:
        return "0-25%"
    if pot_frac <= 0.50:
        return "25-50%"
    if pot_frac <= 0.75:
        return "50-75%"
    if pot_frac <= 1.00:
        return "75-100%"
    if pot_frac <= 1.50:
        return "100-150%"
    return "150%+"


def compute_wetness(board_cards: List[str]) -> float:
    """
    Placeholder wetness measure (0..100). Robust + deterministic but not poker-perfect.
    Returns NaN if board unknown.
    """
    if not board_cards:
        return np.nan
    # basic: more connected/flushy boards are "wetter"
    parsed = [parse_card(c) for c in board_cards if parse_card(c)]
    if len(parsed) < 3:
        return np.nan

    ranks = [p[0] for p in parsed]
    suits = [p[1] for p in parsed]

    # flushiness
    max_suit = max(pd.Series(suits).value_counts().max(), 1)
    flush_score = {1: 0, 2: 25, 3: 55, 4: 80, 5: 95}.get(int(max_suit), 95)

    # connectivity (rough)
    # map ranks to numeric
    def r_to_n(r: str) -> int:
        order = {"A": 14, "K": 13, "Q": 12, "J": 11, "10": 10,
                 "9": 9, "8": 8, "7": 7, "6": 6, "5": 5, "4": 4, "3": 3, "2": 2}
        return order.get(r, 0)

    nums = sorted([r_to_n(r) for r in ranks], reverse=True)
    gaps = []
    for a, b in zip(nums, nums[1:]):
        gaps.append(abs(a - b))
    gap_score = 0
    if gaps:
        # small gaps => wetter
        avg_gap = float(np.mean(gaps))
        if avg_gap <= 1.5:
            gap_score = 60
        elif avg_gap <= 2.5:
            gap_score = 40
        elif avg_gap <= 4.0:
            gap_score = 20
        else:
            gap_score = 5

    # paired boards are slightly drier (fewer straight/flush combos)
    pair_penalty = 0
    if len(set(ranks)) < len(ranks):
        pair_penalty = 10

    wet = 0.6 * flush_score + 0.4 * gap_score - pair_penalty
    return float(np.clip(wet, 0, 100))


def infer_positions(seat_order: List[str], dealer: Optional[str]) -> Dict[str, str]:
    """
    Given seat_order around the table and dealer name, infer basic position labels.
    If dealer missing or not in seat_order, returns {}.
    """
    if not dealer or not seat_order or dealer not in seat_order:
        return {}

    n = len(seat_order)
    di = seat_order.index(dealer)

    # order starting at dealer (button), then clockwise
    rot = seat_order[di:] + seat_order[:di]

    pos_map = {}

    if n == 2:
        pos_map[rot[0]] = "BU/SB"
        pos_map[rot[1]] = "BB"
        return pos_map

    # always define BU/SB/BB first
    pos_map[rot[0]] = "BU"
    pos_map[rot[1]] = "SB"
    pos_map[rot[2]] = "BB"

    # remaining seats
    rest = rot[3:]
    # common naming sets by table size
    # these labels are "good enough" for filtering; can refine later
    name_sets = {
        3: [],
        4: ["UTG"],
        5: ["UTG", "CO"],
        6: ["UTG", "MP", "CO"],
        7: ["UTG", "UTG+1", "MP", "CO"],
        8: ["UTG", "UTG+1", "UTG+2", "MP", "HJ", "CO"],
        9: ["UTG", "UTG+1", "UTG+2", "MP", "MP+1", "HJ", "CO"],
    }
    labels = name_sets.get(n, [])
    if not labels and len(rest) > 0:
        labels = [f"POS{i+1}" for i in range(len(rest))]

    for p, lab in zip(rest, labels):
        pos_map[p] = lab

    return pos_map


# -------------------------
# Regex patterns
# -------------------------
RE_START = re.compile(r"--\s*starting hand\s*#(?P<hn>\d+).*?\(dealer:\s*\"(?P<dealer>[^\"]+)\"\)")
RE_END = re.compile(r"--\s*ending hand\s*#(?P<hn>\d+)\s*--")

RE_STACKS = re.compile(r"Player stacks:\s*(?P<body>.+)$")
RE_STACK_ITEM = re.compile(r"#(?P<seat>\d+)\s*\"(?P<name>[^\"]+)\"\s*\((?P<stack>[0-9]+(?:\.[0-9]+)?)\)")

RE_YOUR_HAND = re.compile(r"Your hand is\s*(?P<c1>[^,]+),\s*(?P<c2>.+)$")

RE_POST_BLIND = re.compile(r"\"(?P<player>[^\"]+)\"\s+posts a\s+(?P<which>small|big)\s+blind of\s+(?P<amt>[0-9]+(?:\.[0-9]+)?)")
RE_CHECK = re.compile(r"\"(?P<player>[^\"]+)\"\s+checks")
RE_FOLD = re.compile(r"\"(?P<player>[^\"]+)\"\s+folds")
RE_CALL = re.compile(r"\"(?P<player>[^\"]+)\"\s+calls\s+(?P<amt>[0-9]+(?:\.[0-9]+)?)")
RE_BET = re.compile(r"\"(?P<player>[^\"]+)\"\s+bets\s+(?P<amt>[0-9]+(?:\.[0-9]+)?)")
RE_RAISE_TO = re.compile(r"\"(?P<player>[^\"]+)\"\s+raises to\s+(?P<amt>[0-9]+(?:\.[0-9]+)?)")

RE_UNCALLED = re.compile(r"Uncalled bet of\s+(?P<amt>[0-9]+(?:\.[0-9]+)?)\s+returned to\s+\"(?P<player>[^\"]+)\"")
RE_COLLECTED = re.compile(r"\"(?P<player>[^\"]+)\"\s+collected\s+(?P<amt>[0-9]+(?:\.[0-9]+)?)\s+from pot")
RE_SHOWS = re.compile(r"\"(?P<player>[^\"]+)\"\s+shows a\s+(?P<c1>[^,]+),\s*(?P<c2>.+?)[\.\,]$")

RE_BOARD_FLOP = re.compile(r"^Flop:\s*(?P<body>.+)$")
RE_BOARD_TURN = re.compile(r"^Turn:\s*(?P<body>.+)$")
RE_BOARD_RIVER = re.compile(r"^River:\s*(?P<body>.+)$")

RE_ALLIN_HINT = re.compile(r"\bgo all in\b", re.IGNORECASE)


@dataclass
class HandState:
    hand_no: int
    dealer: Optional[str] = None
    seat_order: List[str] = field(default_factory=list)
    pos_map: Dict[str, str] = field(default_factory=dict)
    current_street: str = "PREFLOP"
    board: List[str] = field(default_factory=list)
    pot: float = 0.0
    put_in_street: Dict[str, float] = field(default_factory=dict)

    # hero inference
    hero_cards: Optional[Tuple[str, str]] = None
    hero_event_idx: Optional[int] = None  # index into global events list (for updating player)
    lines_seen: int = 0


class PokerLogParser:
    """
    Robust parser for logs that may be chronological or reverse chronological.
    """

    def __init__(self, bb_amt: float = 40.0, sb_amt: float = 20.0, verbose_hands: bool = False):
        self.bb_amt = float(bb_amt)
        self.sb_amt = float(sb_amt)
        self.verbose_hands = bool(verbose_hands)
        self.hero_name: Optional[str] = None

    # ---------- combo ----------
    @staticmethod
    def _to_combo(c1: str, c2: str) -> Optional[str]:
        p1 = parse_card(c1)
        p2 = parse_card(c2)
        if not p1 or not p2:
            return None

        r1, s1 = p1
        r2, s2 = p2
        r1 = normalize_rank(r1)
        r2 = normalize_rank(r2)

        if r1 == r2:
            return f"{r1}{r2}"

        # order by strength using _RANK_ORDER
        i1 = _RANK_INDEX.get(r1, 999)
        i2 = _RANK_INDEX.get(r2, 999)
        suited = (s1 == s2)

        # higher rank first
        if i1 < i2:
            hi, lo = r1, r2
        else:
            hi, lo = r2, r1

        suffix = "s" if suited else "o"
        return f"{hi}{lo}{suffix}"

    # ---------- ordering ----------
    @staticmethod
    def _should_reverse(entries: List[str]) -> bool:
        """
        Detect reverse-chronological log order.
        If the file begins with many 'ending hand' markers before 'starting hand', we reverse.
        """
        first_start = None
        first_end = None
        for i, s in enumerate(entries[:2000]):  # look early
            if first_start is None and RE_START.search(s):
                first_start = i
            if first_end is None and RE_END.search(s):
                first_end = i
            if first_start is not None and first_end is not None:
                break

        if first_end is not None and (first_start is None or first_end < first_start):
            return True

        # fallback: count patterns end->(stuff)->start of same hand in forward direction
        # if many, likely reverse
        end_then_start_same = 0
        last_end = None  # (hn, idx)
        for i, s in enumerate(entries[:5000]):
            me = RE_END.search(s)
            if me:
                last_end = (int(me.group("hn")), i)
                continue
            ms = RE_START.search(s)
            if ms and last_end:
                hn_s = int(ms.group("hn"))
                hn_e, idx_e = last_end
                if hn_s == hn_e and (i - idx_e) < 1000:
                    end_then_start_same += 1
        return end_then_start_same >= 2

    # ---------- parse ----------
    def parse_csv(self, csv_path: str, entry_col: str = "entry") -> Tuple[pd.DataFrame, pd.DataFrame]:
        raw = pd.read_csv(csv_path)
        if entry_col not in raw.columns:
            raise ValueError(f"entry_col='{entry_col}' not found in columns: {raw.columns.tolist()}")

        entries = raw[entry_col].astype(str).tolist()

        reversed_mode = self._should_reverse(entries)
        if reversed_mode:
            entries = list(reversed(entries))
            if self.verbose_hands:
                print("INFO: Detected reverse-order log; reversing entries for chronological parsing.")

        events: List[Dict[str, Any]] = []
        idx_counter = 0

        cur: Optional[HandState] = None

        def add_event(hand_no: int, street: str, rawline: str, player: Optional[str], kind: str,
                      amount: Optional[float] = None, delta_put_in: Optional[float] = None,
                      pot_before: Optional[float] = None, pot_after: Optional[float] = None,
                      show_cards: Optional[Tuple[str, str]] = None,
                      board: Optional[List[str]] = None,
                      position: Optional[str] = None,
                      wetness: Optional[float] = None):
            nonlocal idx_counter
            ev = {
                "hand_no": int(hand_no),
                "street": street,
                "idx": int(idx_counter),
                "raw": rawline,
                "player": player,
                "kind": kind,
                "amount": amount,
                "pot_before": pot_before,
                "pot_after": pot_after,
                "delta_put_in": delta_put_in,
                "pot_frac": np.nan,
                "pot_frac_bucket": None,
                "board": board,
                "wetness": wetness,
                "show_cards": show_cards,
                "position": position,
            }
            idx_counter += 1
            events.append(ev)

        def update_pot_for_put_in(player: str, nominal_amount: float, kind: str, rawline: str):
            """
            Updates pot tracking based on action semantics:
              - bet/call: nominal_amount is delta put in
              - raise_to: nominal_amount is TOTAL for the street (so delta = total - already_put)
              - post blinds: nominal_amount is delta
            """
            if cur is None:
                return

            player = str(player)
            pot_before = float(cur.pot)

            already = float(cur.put_in_street.get(player, 0.0))

            if kind == "raise":
                total = float(nominal_amount)
                delta = max(total - already, 0.0)
                cur.put_in_street[player] = already + delta
            else:
                delta = float(nominal_amount)
                cur.put_in_street[player] = already + delta

            cur.pot = pot_before + delta
            pot_after = float(cur.pot)

            pot_frac = (delta / pot_before) if pot_before > 0 else np.nan

            position = cur.pos_map.get(player, "UNK") if cur else "UNK"

            add_event(
                hand_no=cur.hand_no,
                street=cur.current_street,
                rawline=rawline,
                player=player,
                kind=kind,
                amount=float(nominal_amount),
                delta_put_in=float(delta),
                pot_before=pot_before,
                pot_after=pot_after,
                show_cards=None,
                board=cur.board.copy() if cur.board else None,
                position=position,
                wetness=compute_wetness(cur.board),
            )
            # fill derived pot_frac info
            events[-1]["pot_frac"] = pot_frac
            events[-1]["pot_frac_bucket"] = pot_bucket(pot_frac)

        def finalize_hand():
            if cur is None:
                return

            # if we saw hero cards and know hero name, patch the hole event
            if cur.hero_event_idx is not None and self.hero_name:
                try:
                    events[cur.hero_event_idx]["player"] = self.hero_name
                    if cur.pos_map:
                        events[cur.hero_event_idx]["position"] = cur.pos_map.get(self.hero_name, "UNK")
                except Exception:
                    pass

            if self.verbose_hands:
                # count events in this hand
                evs = [e for e in events if int(e["hand_no"]) == int(cur.hand_no)]
                kinds = pd.Series([e["kind"] for e in evs]).value_counts().to_dict() if evs else {}
                print("=" * 72)
                print(f"HAND #{cur.hand_no} summary")
                print(f"  dealer: {cur.dealer}")
                print(f"  seats: {len(cur.seat_order)} -> {cur.seat_order}")
                if cur.pos_map:
                    print(f"  positions: {cur.pos_map}")
                else:
                    print("  positions: (not inferred; missing stacks or dealer)")
                print(f"  final board: {cur.board}")
                print(f"  final pot (tracked): {cur.pot:.2f}")
                print(f"  lines seen: {cur.lines_seen}")
                print(f"  event counts: {kinds}")
                print("=" * 72)

        # parse line-by-line
        for rawline in entries:
            line = str(rawline).strip()
            if cur is not None:
                cur.lines_seen += 1

            ms = RE_START.search(line)
            if ms:
                # new hand begins; finalize previous
                finalize_hand()

                hn = int(ms.group("hn"))
                dealer = ms.group("dealer")
                if self.verbose_hands:
                    print(f"\n--- START hand #{hn} dealer={dealer} ---")

                cur = HandState(hand_no=hn, dealer=dealer)
                cur.current_street = "PREFLOP"
                cur.board = []
                cur.pot = 0.0
                cur.put_in_street = {}
                continue

            me = RE_END.search(line)
            if me:
                # end of current hand (if matches)
                hn_end = int(me.group("hn"))
                # sometimes "end" marker appears even if we didn't start (should be rare after reversal)
                if cur is None or cur.hand_no != hn_end:
                    # tolerate but do not crash
                    continue
                if self.verbose_hands:
                    print(f"--- END hand #{hn_end} ---\n")
                finalize_hand()
                cur = None
                continue

            # if we haven't started a hand yet, ignore
            if cur is None:
                continue

            # Player stacks (seats)
            mstk = RE_STACKS.search(line)
            if mstk:
                body = mstk.group("body")
                items = RE_STACK_ITEM.findall(body)
                # sort by seat number
                parsed = []
                for seat, name, stack in items:
                    parsed.append((int(seat), name, float(stack)))
                parsed.sort(key=lambda x: x[0])

                cur.seat_order = [name for _, name, _ in parsed]
                cur.pos_map = infer_positions(cur.seat_order, cur.dealer)
                continue

            # Your hand
            myh = RE_YOUR_HAND.search(line)
            if myh:
                c1 = myh.group("c1").strip()
                c2 = myh.group("c2").strip()
                # normalize extracted card substrings
                cards = cards_from_bracket_list(f"{c1} {c2}")
                if len(cards) >= 2:
                    c1n, c2n = cards[0], cards[1]
                else:
                    c1n, c2n = c1, c2

                cur.hero_cards = (c1n, c2n)

                # default player label until we infer who "you" are
                player_label = self.hero_name if self.hero_name else "HERO"

                position = cur.pos_map.get(player_label, "UNK") if cur.pos_map else "UNK"

                add_event(
                    hand_no=cur.hand_no,
                    street=cur.current_street,
                    rawline=line,
                    player=player_label,
                    kind="hole",
                    amount=np.nan,
                    delta_put_in=np.nan,
                    pot_before=float(cur.pot),
                    pot_after=float(cur.pot),
                    show_cards=(c1n, c2n),
                    board=cur.board.copy() if cur.board else None,
                    position=position,
                    wetness=compute_wetness(cur.board),
                )
                cur.hero_event_idx = len(events) - 1
                continue

            # Board lines
            mf = RE_BOARD_FLOP.search(line)
            mt = RE_BOARD_TURN.search(line)
            mr = RE_BOARD_RIVER.search(line)

            if mf:
                cards = cards_from_bracket_list(mf.group("body"))
                cur.current_street = "FLOP"
                cur.put_in_street = {}
                cur.board = cards[:3]
                add_event(
                    hand_no=cur.hand_no,
                    street="FLOP",
                    rawline=line,
                    player=None,
                    kind="board",
                    amount=np.nan,
                    delta_put_in=np.nan,
                    pot_before=float(cur.pot),
                    pot_after=float(cur.pot),
                    show_cards=None,
                    board=cur.board.copy(),
                    position=None,
                    wetness=compute_wetness(cur.board),
                )
                continue

            if mt:
                cards = cards_from_bracket_list(mt.group("body"))
                cur.current_street = "TURN"
                cur.put_in_street = {}
                cur.board = cards[:4]
                add_event(
                    hand_no=cur.hand_no,
                    street="TURN",
                    rawline=line,
                    player=None,
                    kind="board",
                    amount=np.nan,
                    delta_put_in=np.nan,
                    pot_before=float(cur.pot),
                    pot_after=float(cur.pot),
                    show_cards=None,
                    board=cur.board.copy(),
                    position=None,
                    wetness=compute_wetness(cur.board),
                )
                continue

            if mr:
                cards = cards_from_bracket_list(mr.group("body"))
                cur.current_street = "RIVER"
                cur.put_in_street = {}
                cur.board = cards[:5]
                add_event(
                    hand_no=cur.hand_no,
                    street="RIVER",
                    rawline=line,
                    player=None,
                    kind="board",
                    amount=np.nan,
                    delta_put_in=np.nan,
                    pot_before=float(cur.pot),
                    pot_after=float(cur.pot),
                    show_cards=None,
                    board=cur.board.copy(),
                    position=None,
                    wetness=compute_wetness(cur.board),
                )
                continue

            # Uncalled bet returned
            munc = RE_UNCALLED.search(line)
            if munc:
                player = munc.group("player")
                amt = float(munc.group("amt"))
                pot_before = float(cur.pot)
                cur.pot = max(cur.pot - amt, 0.0)
                pot_after = float(cur.pot)
                position = cur.pos_map.get(player, "UNK") if cur.pos_map else "UNK"

                add_event(
                    hand_no=cur.hand_no,
                    street=cur.current_street,
                    rawline=line,
                    player=player,
                    kind="uncalled",
                    amount=amt,
                    delta_put_in=-amt,
                    pot_before=pot_before,
                    pot_after=pot_after,
                    show_cards=None,
                    board=cur.board.copy() if cur.board else None,
                    position=position,
                    wetness=compute_wetness(cur.board),
                )
                continue

            # Collected
            mcol = RE_COLLECTED.search(line)
            if mcol:
                player = mcol.group("player")
                amt = float(mcol.group("amt"))
                position = cur.pos_map.get(player, "UNK") if cur.pos_map else "UNK"
                add_event(
                    hand_no=cur.hand_no,
                    street=cur.current_street,
                    rawline=line,
                    player=player,
                    kind="collected",
                    amount=amt,
                    delta_put_in=np.nan,
                    pot_before=float(cur.pot),
                    pot_after=float(cur.pot),
                    show_cards=None,
                    board=cur.board.copy() if cur.board else None,
                    position=position,
                    wetness=compute_wetness(cur.board),
                )
                continue

            # Shows
            msh = RE_SHOWS.search(line)
            if msh:
                player = msh.group("player")
                c1 = msh.group("c1").strip()
                c2 = msh.group("c2").strip()
                cards = cards_from_bracket_list(f"{c1} {c2}")
                if len(cards) >= 2:
                    c1n, c2n = cards[0], cards[1]
                else:
                    c1n, c2n = c1, c2

                # infer hero name if we have hero cards for this hand and they match
                if cur.hero_cards and not self.hero_name:
                    a = set(cur.hero_cards)
                    b = set([c1n, c2n])
                    if a == b:
                        self.hero_name = player
                        if self.verbose_hands:
                            print(f"INFO: Inferred hero_name = {self.hero_name} (matched Your hand cards).")
                        # patch hole event now
                        if cur.hero_event_idx is not None:
                            events[cur.hero_event_idx]["player"] = self.hero_name
                            if cur.pos_map:
                                events[cur.hero_event_idx]["position"] = cur.pos_map.get(self.hero_name, "UNK")

                position = cur.pos_map.get(player, "UNK") if cur.pos_map else "UNK"
                add_event(
                    hand_no=cur.hand_no,
                    street=cur.current_street,
                    rawline=line,
                    player=player,
                    kind="shows",
                    amount=np.nan,
                    delta_put_in=np.nan,
                    pot_before=float(cur.pot),
                    pot_after=float(cur.pot),
                    show_cards=(c1n, c2n),
                    board=cur.board.copy() if cur.board else None,
                    position=position,
                    wetness=compute_wetness(cur.board),
                )
                continue

            # Posts blinds
            mpb = RE_POST_BLIND.search(line)
            if mpb:
                player = mpb.group("player")
                which = mpb.group("which")
                amt = float(mpb.group("amt"))
                kind = "post_sb" if which == "small" else "post_bb"
                update_pot_for_put_in(player, amt, kind, line)
                continue

            # Checks/folds
            mch = RE_CHECK.search(line)
            if mch:
                player = mch.group("player")
                position = cur.pos_map.get(player, "UNK") if cur.pos_map else "UNK"
                add_event(
                    hand_no=cur.hand_no,
                    street=cur.current_street,
                    rawline=line,
                    player=player,
                    kind="check",
                    amount=np.nan,
                    delta_put_in=0.0,
                    pot_before=float(cur.pot),
                    pot_after=float(cur.pot),
                    show_cards=None,
                    board=cur.board.copy() if cur.board else None,
                    position=position,
                    wetness=compute_wetness(cur.board),
                )
                continue

            mfd = RE_FOLD.search(line)
            if mfd:
                player = mfd.group("player")
                position = cur.pos_map.get(player, "UNK") if cur.pos_map else "UNK"
                add_event(
                    hand_no=cur.hand_no,
                    street=cur.current_street,
                    rawline=line,
                    player=player,
                    kind="fold",
                    amount=np.nan,
                    delta_put_in=0.0,
                    pot_before=float(cur.pot),
                    pot_after=float(cur.pot),
                    show_cards=None,
                    board=cur.board.copy() if cur.board else None,
                    position=position,
                    wetness=compute_wetness(cur.board),
                )
                continue

            # Calls/bets/raises
            mcall = RE_CALL.search(line)
            if mcall:
                player = mcall.group("player")
                amt = float(mcall.group("amt"))
                kind = "call"
                if RE_ALLIN_HINT.search(line):
                    kind = "allin"
                update_pot_for_put_in(player, amt, kind, line)
                continue

            mbet = RE_BET.search(line)
            if mbet:
                player = mbet.group("player")
                amt = float(mbet.group("amt"))
                kind = "bet"
                if RE_ALLIN_HINT.search(line):
                    kind = "allin"
                update_pot_for_put_in(player, amt, kind, line)
                continue

            mraise = RE_RAISE_TO.search(line)
            if mraise:
                player = mraise.group("player")
                amt = float(mraise.group("amt"))
                kind = "raise"
                if RE_ALLIN_HINT.search(line):
                    kind = "allin"
                update_pot_for_put_in(player, amt, kind, line)
                continue

            # else: ignore meta lines like run it twice, joins/leaves, undealt cards, etc.

        # finalize if file ended mid-hand
        finalize_hand()

        events_df = pd.DataFrame(events)

        # add n_players if possible (from seat_order is not stored in events; compute from player column)
        if not events_df.empty:
            events_df["n_players"] = (
                events_df.groupby("hand_no")["player"]
                .transform(lambda s: s.dropna().nunique())
                .astype(int)
            )
        else:
            events_df["n_players"] = pd.Series(dtype=int)

        # normalize dtypes
        if "player" in events_df.columns:
            events_df["player"] = events_df["player"].astype(object)

        # summary_df intentionally empty; run_analysis computes its own
        summary_df = pd.DataFrame()
        return events_df, summary_df
