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

Per (read, candidate) the cheaper side determines the read's minimum-cost
choice.  For a threshold t on mismatch count the read is then in one of:
  * infeasible: cheapest-side mismatch count > t;
  * forced: exactly one side keeps the minimum cost (or, on an equal-cost tie,
    the second side needs more than t mismatches);
  * flexible: both sides keep the minimum cost and stay within t.

Ties between full solutions are broken deterministically on
(max mismatches, haplotype string lexicographic on locus order, assignment
tuple lexicographic on read order); the first two *distinct* optimum solutions
are returned, which also gives the unique/ambiguous decision.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

MIN_LOCI = 8
MAX_LOCI = 18
MIN_READS = 10
MAX_READS = 36
MAX_COST = 10**9
MIN_GROUP_SIZE = 2

# Candidate haplotypes are processed in chunks to bound peak memory.
CHUNK = 8192

# flex_kind codes
FLEX_NEVER = 0       # one side is strictly cheaper: the read can never switch
FLEX_SYMMETRIC = 1   # equal cost, equal mismatch count: free at c_forced
FLEX_ASYMMETRIC = 2  # equal cost, unequal count: free only at the larger count


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
    """Evaluate a block of normalised candidate haplotypes.

    g encodes H0 on loci 0..n-3 as low bits; loci n-2 and n-1 are 0.

    Returns per-(read, candidate) arrays describing the cheapest assignment:
      grp_forced : side feasible at the smallest mismatch count
      c_forced   : mismatch count on that side
      c_flex     : mismatch count at which the other side becomes feasible
                   while keeping the minimum cost (c_forced if symmetric,
                   larger count if asymmetric, +inf if never flexible)
      flex_kind  : FLEX_* code
      cost_min   : minimum mismatch cost for the read
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

    cheaper0 = cost0 < cost1
    equal = cost0 == cost1
    fewer0 = cnt0 < cnt1
    fewer1 = cnt1 < cnt0

    grp_forced = np.where(cheaper0 | (equal & fewer0), 0, 1)
    c0_best = cheaper0 | (equal & ~fewer1)   # side 0 is (a) cheapest choice
    c_forced = np.where(c0_best, cnt0, cnt1)
    cost_min = np.minimum(cost0, cost1)

    flex_kind = np.full((m, B), FLEX_NEVER, dtype=np.int64)
    c_flex = np.full((m, B), np.iinfo(np.int64).max, dtype=np.int64)

    sym = equal & (cnt0 == cnt1)
    asym = equal & ~sym
    flex_kind[sym] = FLEX_SYMMETRIC
    c_flex[sym] = cnt0[sym]
    flex_kind[asym] = FLEX_ASYMMETRIC
    c_flex[asym] = np.maximum(cnt0, cnt1)[asym]

    return grp_forced, c_forced, c_flex, flex_kind, cost_min


def _balance_possible(n0_forced, k_flex, m: int) -> np.ndarray:
    """Can x flexible reads be put in group 0 so both groups have >= 2 reads?"""
    x_lo = np.maximum(0, MIN_GROUP_SIZE - n0_forced)
    x_hi = np.minimum(k_flex, m - MIN_GROUP_SIZE - n0_forced)
    return x_lo <= x_hi


def _scan_candidates(n, obs, mask, costs, limits, lengths):
    """First pass: feasibility at each read's own budget and minimum cost."""
    m = obs.shape[0]
    P = 1 << (n - 2)
    feasible = np.zeros(P, dtype=bool)
    totals = np.full(P, np.iinfo(np.int64).max, dtype=np.int64)
    size_ok_all = np.zeros(P, dtype=bool)
    budget_ok_all = np.zeros(P, dtype=bool)

    lim = limits[:, None]
    for start in range(0, P, CHUNK):
        g = np.arange(start, min(start + CHUNK, P), dtype=np.int64)
        grp, c_forced, c_flex, flex_kind, cost_min = _candidate_columns(
            g, obs, mask, costs, lengths)

        budget_ok = (c_forced <= lim).all(axis=0)

        flex = ((flex_kind == FLEX_SYMMETRIC) & (c_forced <= lim)) | \
               ((flex_kind == FLEX_ASYMMETRIC) & (c_flex <= lim))
        n0_forced = ((grp == 0) & ~flex).sum(axis=0)
        k_flex = flex.sum(axis=0)
        size_ok = _balance_possible(n0_forced, k_flex, m)

        ok = budget_ok & size_ok
        feasible[g] = ok
        size_ok_all[g] = size_ok
        budget_ok_all[g] = budget_ok
        totals[g] = cost_min.sum(axis=0)

    return feasible, totals, size_ok_all, budget_ok_all


def _reverse_bits(g: np.ndarray, n: int) -> np.ndarray:
    """Numeric order then matches haplotype-string (locus 0 first) order."""
    rev = np.zeros_like(g)
    for j in range(n):
        rev |= ((g >> j) & 1) << (n - 1 - j)
    return rev


def _threshold_columns(indices, obs, mask, costs, lengths, m: int):
    """Second pass: min feasible mismatch threshold t per candidate.

    Also returns the forced-group template / flexible flags at that threshold
    (one column per candidate), ready for assignment enumeration.
    """
    g = indices.astype(np.int64)
    grp, c_forced, c_flex, flex_kind, _ = _candidate_columns(
        g, obs, mask, costs, lengths)
    k = g.shape[0]
    max_t = int(lengths.max())

    t_star = np.full(k, max_t + 1, dtype=np.int64)
    for t in range(max_t + 1):
        infeasible = (c_forced > t).any(axis=0)
        flex = (flex_kind != FLEX_NEVER) & (c_flex <= t)
        n0_forced = ((grp == 0) & ~flex).sum(axis=0)
        k_flex = flex.sum(axis=0)
        ok = ~infeasible & _balance_possible(n0_forced, k_flex, m) & (t_star > max_t)
        t_star[ok] = t
        if bool((t_star <= max_t).all()):
            break

    # Recompute flex / forced templates exactly at t_star for each candidate.
    tt = t_star[None, :]
    flex = (flex_kind != FLEX_NEVER) & (c_flex <= tt)
    forced = ~flex
    return t_star, grp, forced, flex


# ---------------------------------------------------------------------------
# Deterministic enumeration of the first two distinct optimum assignments
# ---------------------------------------------------------------------------

def _ways(rem: int, lo: int, hi: int) -> int:
    """Number of ways to choose x of rem flexible reads with lo <= x <= hi."""
    lo = max(lo, 0)
    hi = min(hi, rem)
    if lo > hi:
        return 0
    return sum(math.comb(rem, x) for x in range(lo, hi + 1))


def _kth_assignment(template: list[int], flex_pos: list[int], n0_forced: int,
                    m: int, rank: int) -> tuple[int, ...]:
    """rank-th lexicographically smallest valid assignment (rank 1-based)."""
    assign = list(template)
    n0 = n0_forced
    rem = len(flex_pos)
    for fp in flex_pos:
        rem -= 1
        lo = MIN_GROUP_SIZE - n0                 # flex reads still placed as 0
        hi = m - MIN_GROUP_SIZE - n0
        w0 = _ways(rem, lo, hi)
        if rank <= w0:
            assign[fp] = 0
        else:
            rank -= w0
            assign[fp] = 1
            n0 += 1
    return tuple(assign)


def _column_parts(grp_col, forced_col, flex_col) -> tuple[list[int], list[int], int]:
    m = grp_col.shape[0]
    template = [-1 if flex_col[i] else int(grp_col[i]) for i in range(m)]
    flex_pos = [i for i in range(m) if flex_col[i]]
    n0_forced = sum(1 for v in template if v == 0)
    return template, flex_pos, n0_forced


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
                f"no candidate haplotype pair keeps every read within its "
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

    # Second objective: minimal maximum per-read mismatch count.
    t_star, grp, forced, flex = _threshold_columns(
        best_idx, obs, mask, costs, lengths, m)
    t_min = int(t_star.min())
    pool = best_idx[t_star == t_min]
    pool = pool[np.argsort(_reverse_bits(pool, n), kind="stable")]

    g_a = int(pool[0])
    col_a = int(np.searchsorted(best_idx, g_a))
    template_a, flex_pos_a, n0_forced_a = _column_parts(
        grp[:, col_a], forced[:, col_a], flex[:, col_a])
    kf_a = len(flex_pos_a)
    count_a = _ways(kf_a, MIN_GROUP_SIZE - n0_forced_a,
                    m - MIN_GROUP_SIZE - n0_forced_a)
    assign_a1 = _kth_assignment(template_a, flex_pos_a, n0_forced_a, m, 1)

    second = None
    if count_a >= 2:
        second = (g_a, _kth_assignment(template_a, flex_pos_a, n0_forced_a, m, 2))
    elif pool.shape[0] > 1:
        g_b = int(pool[1])
        col_b = int(np.searchsorted(best_idx, g_b))
        template_b, flex_pos_b, n0_forced_b = _column_parts(
            grp[:, col_b], forced[:, col_b], flex[:, col_b])
        assign_b1 = _kth_assignment(template_b, flex_pos_b, n0_forced_b, m, 1)
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
