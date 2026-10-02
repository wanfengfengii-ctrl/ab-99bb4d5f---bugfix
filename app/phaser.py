"""Ancient-DNA haplotype phasing solver.

Given binary observations from degraded molecule reads (each read covers one
contiguous interval of ordered biallelic loci and carries per-locus positive
mismatch costs plus an allowed mismatch-count budget), jointly reconstruct a
pair of complementary haplotypes and assign every read uniquely to one of the
two homologs.

Objective (lexicographic):
  1. minimise total mismatch cost over all reads;
  2. among those solutions, minimise the maximum per-read mismatch count.

Swapping the two homolog labels (H, a) <-> (~H, ~a) is the same solution, so
haplotypes are normalised: the representative H0 has its two most significant
bits (loci n-1 and n-2) equal to 0.

Per (read, candidate) *both* sides are evaluated:
  * A read is budget-feasible on a side if its mismatch count there is within
    its 'max_mismatches' budget.  The cheapest side is never assumed feasible:
    a read whose cheap side needs two mismatches but whose expensive side
    needs one is perfectly feasible (at the higher cost) under a one-mismatch
    budget.
  * Reads start on their cheapest feasible side; moving a read to the other
    feasible side costs a non-negative premium (zero when both sides tie).
    Reads may be switched -- paying a premium when needed -- to reach the
    required MIN_GROUP_SIZE reads on each homolog, and to minimise the maximum
    per-read mismatch count once total cost is fixed.

For a fixed candidate the minimum total cost of a balanced assignment (each
homolog >= MIN_GROUP_SIZE reads) is the base cost plus the cheapest switch
premiums needed in each direction; MIN_GROUP_SIZE is 2, so at most two
premiums per direction are ever required.  The optimal mismatch threshold is
found by scanning t = 0, 1, ... with the same vectorised calculation restricted
to sides feasible within t mismatches.

Optimal *assignments* for the (at most two) winning candidates are enumerated
with a small suffix DP over reads (m <= 36): it returns the lexicographically
smallest optimal assignment, decides whether another distinct optimal
assignment exists, and if so returns its immediate lexicographic successor.

Ties between full solutions are broken deterministically on
(max mismatches, haplotype string lexicographic on locus order, assignment
tuple lexicographic on read order); the first two *distinct* optimum solutions
are returned, which also gives the unique/ambiguous decision.
"""

from __future__ import annotations

from typing import Any

import numpy as np
MIN_LOCI = 8
MAX_LOCI = 18
MIN_READS = 10
MAX_READS = 36
MAX_COST = 10**9
MIN_GROUP_SIZE = 2
INF = np.iinfo(np.int64).max

# Sentinel "switch impossible" premium.  Every real total is at most
# 36 reads * 18 loci * 10**9 < 10**12, so 10**15 is safely larger.
BIG = 10**15

# Candidate haplotypes are processed in chunks to bound peak memory.
CHUNK = 8192


class PhaseError(Exception):
    """Business-level error with a stable machine code."""

    def __init__(self, code: str, message: str, status_code: int = 422):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Input validation and array construction
# ---------------------------------------------------------------------------

def _require(cond: bool, code: str, message: str) -> None:
    if not cond:
        raise PhaseError(code, message)


def parse_input(payload: Any) -> tuple[int, list[dict[str, Any]]]:
    _require(isinstance(payload, dict), "VALIDATION_ERROR",
             "request body must be a JSON object")

    n = payload.get("loci")
    _require(
        isinstance(n, int) and not isinstance(n, bool) and MIN_LOCI <= n <= MAX_LOCI,
        "VALIDATION_ERROR",
        f"'loci' must be an integer in [{MIN_LOCI}, {MAX_LOCI}]",
    )

    raw_reads = payload.get("reads")
    _require(
        isinstance(raw_reads, list)
        and MIN_READS <= len(raw_reads) <= MAX_READS,
        "VALIDATION_ERROR",
        f"'reads' must contain between {MIN_READS} and {MAX_READS} entries",
    )

    reads: list[dict[str, Any]] = []
    seen: set[tuple[tuple[int, ...], str]] = set()
    for idx, rr in enumerate(raw_reads):
        _require(isinstance(rr, dict), "VALIDATION_ERROR",
                 f"read[{idx}] must be an object")
        where = f"read[{idx}]"

        positions = rr.get("positions")
        _require(
            isinstance(positions, list) and len(positions) > 0
            and all(isinstance(p, int) and not isinstance(p, bool) for p in positions),
            "VALIDATION_ERROR",
            f"{where}: 'positions' must be a non-empty list of integers",
        )
        _require(
            all(0 <= p < n for p in positions),
            "VALIDATION_ERROR",
            f"{where}: every position must lie in [0, {n - 1}]",
        )
        _require(
            all(positions[j] < positions[j + 1] for j in range(len(positions) - 1)),
            "VALIDATION_ERROR",
            f"{where}: 'positions' must be strictly increasing",
        )
        _require(
            all(positions[j + 1] == positions[j] + 1
                for j in range(len(positions) - 1)),
            "READ_NOT_CONTIGUOUS",
            f"{where}: a read must cover one contiguous interval "
            f"(got gap in {positions})",
        )

        L = len(positions)
        obs = rr.get("observations")
        _require(
            isinstance(obs, str) and len(obs) == L,
            "VALIDATION_ERROR",
            f"{where}: 'observations' must be a string of length {L} of 0/1",
        )
        _require(
            all(ch in "01" for ch in obs),
            "VALIDATION_ERROR",
            f"{where}: 'observations' may only contain '0' and '1'",
        )

        costs = rr.get("mismatch_costs")
        _require(
            isinstance(costs, list) and len(costs) == L,
            "VALIDATION_ERROR",
            f"{where}: 'mismatch_costs' must be a list of {L} positive integers",
        )
        _require(
            all(isinstance(c, int) and not isinstance(c, bool) and 1 <= c <= MAX_COST
                for c in costs),
            "VALIDATION_ERROR",
            f"{where}: every mismatch cost must be a positive integer <= {MAX_COST}",
        )

        limit = rr.get("max_mismatches")
        _require(
            isinstance(limit, int) and not isinstance(limit, bool) and 0 <= limit <= n,
            "VALIDATION_ERROR",
            f"{where}: 'max_mismatches' must be an integer in [0, {n}]",
        )

        fingerprint = (tuple(positions), obs)
        _require(
            fingerprint not in seen,
            "DUPLICATE_READ",
            f"{where}: duplicate read with same interval and observations",
        )
        seen.add(fingerprint)

        reads.append(
            {"positions": list(positions), "observations": obs,
             "costs": list(costs), "limit": limit}
        )

    return n, reads


def _build_arrays(n: int, reads: list[dict[str, Any]]):
    m = len(reads)
    obs = np.zeros((m, n), dtype=np.int64)
    mask = np.zeros((m, n), dtype=bool)
    costs = np.zeros((m, n), dtype=np.int64)
    limits = np.zeros(m, dtype=np.int64)
    lengths = np.zeros(m, dtype=np.int64)
    for i, r in enumerate(reads):
        ps = r["positions"]
        lengths[i] = len(ps)
        limits[i] = r["limit"]
        for j, p in enumerate(ps):
            mask[i, p] = True
            obs[i, p] = int(r["observations"][j])
            costs[i, p] = r["costs"][j]
    return obs, mask, costs, limits, lengths


# ---------------------------------------------------------------------------
# Vectorised candidate evaluation
# ---------------------------------------------------------------------------

def _candidate_columns(g: np.ndarray, obs, mask, costs, lengths):
    """Mismatch counts and costs on both sides for a block of candidates.

    g encodes H0 on loci 0..n-3 as low bits; loci n-2 and n-1 are 0.

    Returns cnt0/cnt1 (mismatch counts) and cost0/cost1 (mismatch costs) of
    every read against H0 / H1, each with shape (m, B).
    """
    m, n = obs.shape
    B = g.shape[0]

    h0 = np.zeros((B, n), dtype=np.int64)
    for j in range(n - 2):
        h0[:, j] = (g >> j) & 1

    # disagrees with H0 at covered loci
    x = ((obs[:, None, :] ^ h0[None, :, :]) & mask[:, None, :]).astype(np.int64)

    cnt0 = x.sum(axis=2)                                   # (m, B)
    cost0 = np.einsum("mbn,mn->mb", x, costs)
    read_totals = (costs * mask).sum(axis=1)
    cnt1 = lengths[:, None] - cnt0
    cost1 = read_totals[:, None] - cost0
    return cnt0, cnt1, cost0, cost1


def _block_eval(cost0, cost1, allow0, allow1, m: int):
    """Min balanced assignment cost per candidate given allowed sides.

    ``allow0`` / ``allow1`` mark which (read, candidate) sides may be used
    (within the read's own budget, or additionally within a mismatch
    threshold).  Per candidate:

      * budget feasible iff every read has at least one allowed side;
      * each read starts on its cheapest allowed side (ties -> side 0);
      * balance requires moving need0 / need1 reads into group 0 / group 1;
        only reads whose other side is also allowed can move, paying the cost
        difference; the cheapest premiums win.

    Returns budget_ok, balance_ok and the minimum balanced total per column
    (INF where no balanced assignment exists).
    """
    budget_ok = (allow0 | allow1).all(axis=0)

    choose0 = allow0 & (~allow1 | (cost0 <= cost1))
    base_cost = np.where(choose0, cost0, cost1).sum(axis=0)
    n0_base = choose0.sum(axis=0)
    need0 = np.maximum(0, MIN_GROUP_SIZE - n0_base)
    need1 = np.maximum(0, MIN_GROUP_SIZE - (m - n0_base))

    # Premium of switching a read from its base side to the other one; reads
    # whose other side is disallowed carry the BIG sentinel.
    prem_to0 = np.where(~choose0 & allow0 & allow1, cost0 - cost1, BIG)
    prem_to1 = np.where(choose0 & allow0 & allow1, cost1 - cost0, BIG)
    avail0 = (prem_to0 < BIG).sum(axis=0)
    avail1 = (prem_to1 < BIG).sum(axis=0)
    sort0 = np.sort(prem_to0, axis=0)
    sort1 = np.sort(prem_to1, axis=0)

    # need0 / need1 are at most MIN_GROUP_SIZE (2).
    extra0 = np.where(
        need0 == 0, 0,
        np.where(need0 == 1, sort0[0], sort0[0] + sort0[1]))
    extra1 = np.where(
        need1 == 0, 0,
        np.where(need1 == 1, sort1[0], sort1[0] + sort1[1]))

    balance_ok = (avail0 >= need0) & (avail1 >= need1)
    total = np.where(budget_ok & balance_ok,
                     base_cost + extra0 + extra1, INF)
    return budget_ok, balance_ok, total


def _scan_candidates(n, obs, mask, costs, limits, lengths):
    """First pass: budget/balance feasibility and minimum cost per candidate.

    budget_ok : every read has at least one side within its own mismatch budget
    size_ok   : a balanced split exists at all (both sides physically usable,
                budgets ignored).  With m >= 2 * MIN_GROUP_SIZE this holds for
                every candidate, but it is tracked explicitly so the error
                classification can tell a pure budget failure apart from a
                joint failure.
    feasible  : a balanced split exists using only budget-feasible sides
    """
    P = 1 << (n - 2)
    totals = np.full(P, INF, dtype=np.int64)
    size_ok_all = np.zeros(P, dtype=bool)
    budget_ok_all = np.zeros(P, dtype=bool)
    lim = limits[:, None]
    any_side = np.ones((obs.shape[0], 1), dtype=bool)

    for start in range(0, P, CHUNK):
        idx = np.arange(start, min(start + CHUNK, P), dtype=np.int64)
        cnt0, cnt1, cost0, cost1 = _candidate_columns(
            idx, obs, mask, costs, lengths)
        allow0 = cnt0 <= lim
        allow1 = cnt1 <= lim
        budget_ok, balance_ok, total = _block_eval(
            cost0, cost1, allow0, allow1, obs.shape[0])
        _, balance_any, _ = _block_eval(
            cost0, cost1, any_side, any_side, obs.shape[0])
        budget_ok_all[idx] = budget_ok
        size_ok_all[idx] = balance_any
        totals[idx] = total

    feasible = totals < INF
    return feasible, totals, size_ok_all, budget_ok_all


def _reverse_bits(g: np.ndarray, n: int) -> np.ndarray:
    """Numeric order then matches haplotype-string (locus 0 first) order."""
    rev = np.zeros_like(g)
    for j in range(n):
        rev |= ((g >> j) & 1) << (n - 1 - j)
    return rev


def _threshold_scan(indices, obs, mask, costs, limits, lengths, min_total):
    """Smallest feasible mismatch threshold per minimum-cost candidate.

    A candidate keeps global minimum total ``min_total`` at threshold t iff
    every read has a side within min(t, its budget) and the minimum balanced
    cost over those sides still equals ``min_total``.  Fully vectorised over
    candidates; t only ranges over read lengths (<= 18).
    """
    k = indices.shape[0]
    max_t = int(lengths.max())
    lim = limits[:, None]
    cnt0, cnt1, cost0, cost1 = _candidate_columns(
        indices.astype(np.int64), obs, mask, costs, lengths)

    t_star = np.full(k, max_t + 1, dtype=np.int64)
    for t in range(max_t + 1):
        pending = t_star > max_t
        if not bool(pending.any()):
            break
        allow0 = (cnt0 <= t) & (cnt0 <= lim)
        allow1 = (cnt1 <= t) & (cnt1 <= lim)
        _, _, total = _block_eval(cost0, cost1, allow0, allow1, obs.shape[0])
        hit = pending & (total == min_total)
        t_star[hit] = t
    return t_star, cnt0, cnt1, cost0, cost1


# ---------------------------------------------------------------------------
# Optimal-assignment enumeration for one winning candidate
# ---------------------------------------------------------------------------

def _enumerate_optimal(cnt0, cnt1, cost0, cost1, limits, m: int,
                       t: int) -> tuple[tuple[int, ...], tuple[int, ...] | None, int]:
    """Optimal assignments for one candidate at its optimal threshold t.

    Suffix DP over reads: suffix[i][z] is the minimum cost of placing reads
    i..m-1 with exactly z of them on side 0, using only sides with at most t
    mismatches (and within each read's own budget).  Counts how many
    assignments attain each suffix minimum (capped at 2).

    Returns (best assignment, second distinct assignment or None, number of
    distinct optimal assignments capped at 2).
    """
    # Per-read feasible side costs (INF when the side is not allowed).
    per_read = [
        (int(cost0[i]) if (int(cnt0[i]) <= t
                           and int(cnt0[i]) <= int(limits[i])) else INF,
         int(cost1[i]) if (int(cnt1[i]) <= t
                           and int(cnt1[i]) <= int(limits[i])) else INF)
        for i in range(m)
    ]

    # suffix_cost[i][z], suffix_ways[i][z] (ways capped at 2).
    sc = [[INF] * (m + 1) for _ in range(m + 1)]
    sw = [[0] * (m + 1) for _ in range(m + 1)]
    sc[m][0] = 0
    sw[m][0] = 1
    for i in range(m - 1, -1, -1):
        a0, a1 = per_read[i]
        for z in range(m - i + 1):
            best = INF
            ways = 0
            if z > 0 and a0 < INF and sc[i + 1][z - 1] < INF:
                best = a0 + sc[i + 1][z - 1]
                ways = sw[i + 1][z - 1]
            if a1 < INF and sc[i + 1][z] < INF:
                v = a1 + sc[i + 1][z]
                if v < best:
                    best, ways = v, sw[i + 1][z]
                elif v == best:
                    ways = min(2, ways + sw[i + 1][z])
            sc[i][z] = best
            sw[i][z] = ways

    best = min(sc[0][z] for z in range(MIN_GROUP_SIZE, m - MIN_GROUP_SIZE + 1))
    total_ways = min(2, sum(
        sw[0][z] for z in range(MIN_GROUP_SIZE, m - MIN_GROUP_SIZE + 1)
        if sc[0][z] == best))

    def suffix_min(i: int, zlo: int, zhi: int) -> int:
        """Minimum suffix cost for reads i.. with zero-count in [zlo, zhi]."""
        zlo = max(zlo, 0)
        zhi = min(zhi, m - i)
        if zlo > zhi:
            return INF
        return min(sc[i][z] for z in range(zlo, zhi + 1))

    def can_choose(i: int, k: int, spent: int, side: int, best_total: int) -> bool:
        """After choosing `side` for read i, can the rest still reach best?"""
        k2 = k + (1 if side == 0 else 0)
        rem = m - i - 1
        zlo = MIN_GROUP_SIZE - k2
        zhi = m - MIN_GROUP_SIZE - k2
        add = per_read[i][side]
        if add >= INF or spent + add > best_total:
            return False
        return suffix_min(i + 1, zlo, zhi) == best_total - spent - add

    def lex_min_from(i: int, k: int, spent: int, prefix: list[int],
                     best_total: int) -> tuple[int, ...]:
        """Lexicographically smallest optimal completion from read i on."""
        out = list(prefix)
        for j in range(i, m):
            picked = 0 if can_choose(j, k, spent, 0, best_total) else 1
            out.append(picked)
            spent += per_read[j][picked]
            k += 1 if picked == 0 else 0
        return tuple(out)

    first = lex_min_from(0, 0, 0, [], best)

    # Immediate lexicographic successor: rightmost 0 in `first` that can flip
    # to 1 while still completing optimally; the suffix becomes lex-min.
    second = None
    second_state = None
    k0 = 0
    spent0 = 0
    for i in range(m):
        if first[i] == 0 and can_choose(i, k0, spent0, 1, best):
            second_state = (i, k0, spent0)
        k0 += 1 if first[i] == 0 else 0
        spent0 += per_read[i][first[i]]
    if total_ways >= 2 and second_state is not None:
        i, k, spent = second_state
        second = lex_min_from(
            i + 1, k, spent + per_read[i][1],
            list(first[:i]) + [1], best)

    return first, second, total_ways


# ---------------------------------------------------------------------------
# Response formatting
# ---------------------------------------------------------------------------

def _haplotype_strings(g: int, n: int) -> tuple[str, str]:
    h0 = "".join(str((g >> j) & 1) for j in range(n))
    h1 = "".join("1" if ch == "0" else "0" for ch in h0)
    return h0, h1


def _format_solution(rank: int, g: int, assignment: tuple[int, ...], n: int,
                     reads: list[dict[str, Any]], total: int, t_max: int) -> dict[str, Any]:
    h0, h1 = _haplotype_strings(g, n)
    hap = (h0, h1)
    evidence = []
    sizes = [0, 0]
    for i, r in enumerate(reads):
        side = assignment[i]
        sizes[side] += 1
        mismatches = []
        for j, p in enumerate(r["positions"]):
            observed = r["observations"][j]
            expected = hap[side][p]
            if observed != expected:
                mismatches.append({
                    "position": p,
                    "observed": observed,
                    "expected": expected,
                    "cost": r["costs"][j],
                })
        evidence.append({
            "read_id": i,
            "group": side,
            "positions": r["positions"],
            "observations": r["observations"],
            "mismatch_count": len(mismatches),
            "mismatch_cost": sum(mm["cost"] for mm in mismatches),
            "mismatches": mismatches,
            "max_mismatches_allowed": r["limit"],
            "within_mismatch_limit": len(mismatches) <= r["limit"],
        })
    return {
        "solution_rank": rank,
        "haplotypes": {"group_0": h0, "group_1": h1},
        "assignments": evidence,
        "group_sizes": sizes,
        "total_mismatch_cost": int(total),
        "max_mismatches_per_read": int(t_max),
    }


def _no_solution_reason(size_ok: np.ndarray, budget_ok: np.ndarray) -> tuple[str, str]:
    if not size_ok.any() and not budget_ok.any():
        return ("INFEASIBLE",
                "no candidate haplotype pair keeps every read within its "
                "mismatch budget and admits at least "
                f"{MIN_GROUP_SIZE} reads per homolog")
    if not budget_ok.any():
        return ("MISMATCH_BUDGET_EXCEEDED",
                "for every candidate haplotype pair at least one read needs more "
                "mismatches than its 'max_mismatches' allows")
    return ("INFEASIBLE_GROUP_BALANCE",
            "reads within budget can only be placed so that one homolog would "
            f"have fewer than {MIN_GROUP_SIZE} reads, for every candidate pair")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def solve(payload: Any) -> dict[str, Any]:
    n, reads = parse_input(payload)
    m = len(reads)
    obs, mask, costs, limits, lengths = _build_arrays(n, reads)

    feasible, totals, size_ok, budget_ok = _scan_candidates(
        n, obs, mask, costs, limits, lengths)

    if not feasible.any():
        code, message = _no_solution_reason(size_ok, budget_ok)
        raise PhaseError(code, message)

    best_total = int(totals[feasible].min())
    best_idx = np.nonzero(feasible & (totals == best_total))[0]

    # Second objective: minimal maximum per-read mismatch count among the
    # minimum-cost balanced solutions.
    t_star, cnt0, cnt1, cost0, cost1 = _threshold_scan(
        best_idx, obs, mask, costs, limits, lengths, best_total)
    t_min = int(t_star.min())
    pool = best_idx[t_star == t_min]
    pool = pool[np.argsort(_reverse_bits(pool, n), kind="stable")]

    def column_of(g: int) -> int:
        return int(np.searchsorted(best_idx, g))

    def enumerate_candidate(g: int):
        col = column_of(g)
        return _enumerate_optimal(
            cnt0[:, col], cnt1[:, col], cost0[:, col], cost1[:, col],
            limits, m, int(t_star[col]))

    g_a = int(pool[0])
    assign_a1, assign_a2, ways_a = enumerate_candidate(g_a)

    second = None
    if assign_a2 is not None:
        second = (g_a, assign_a2)
    elif pool.shape[0] > 1:
        g_b = int(pool[1])
        assign_b1, _, _ = enumerate_candidate(g_b)
        second = (g_b, assign_b1)

    solutions = [_format_solution(1, g_a, assign_a1, n, reads, best_total, t_min)]
    status = "unique"
    if second is not None:
        status = "ambiguous"
        solutions.append(
            _format_solution(2, second[0], second[1], n, reads, best_total, t_min))

    return {
        "status": status,
        "loci": n,
        "num_reads": m,
        "objective": {
            "total_mismatch_cost": best_total,
            "max_mismatches_per_read": t_min,
        },
        "solutions": solutions,
        "tie_order": ["max_mismatches_per_read",
                      "haplotype_string_locus_order",
                      "assignment_tuple_read_order"],
    }
