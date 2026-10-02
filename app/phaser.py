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

Per (read, candidate) both homolog sides have a mismatch count and cost.
A side is *feasible* when its mismatch count is within the read's budget;
a read may be:
  * infeasible: neither side keeps within its mismatch budget;
  * one-sided: only one side is feasible (it may be the costlier side --
    paying its premium is mandatory for this read);
  * two-sided: both sides are feasible; choosing the costlier side is only
    ever needed to satisfy the per-group minimum size, or on an equal-cost
    tie where it can trade a different mismatch count.

The minimum feasible total cost for a candidate is therefore a small
two-label assignment problem (only the number of reads on each homolog is
constrained), solved for all candidates jointly with a vectorised min-cost
DP whose secondary value tracks the minimum maximum per-read mismatch
count.  Reconstruction of the winning candidate(s) (saturated enumeration
of optimal assignments for the unique/ambiguous decision and deterministic
ranking) uses an exact scalar forward/backward DP over reads and group
sizes.

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

# Candidate haplotypes are processed in chunks to bound peak memory.
CHUNK = 8192

# Sentinels for unreachable dynamic-programming states.
_INF = np.iinfo(np.int64).max
_INF_T = np.int64(10**9)


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

    Returns per-(read, candidate) arrays for both homolog sides:
      cnt0, cost0 : mismatch count / cost when the read is assigned to H0
      cnt1, cost1 : mismatch count / cost when assigned to the complement
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
    return cnt0, cost0, cnt1, cost1


def _feasible_sides(cnt0, cnt1, lim) -> tuple[np.ndarray, np.ndarray]:
    """Booleans (f0, f1): side k keeps within the read's mismatch budget."""
    return cnt0 <= lim, cnt1 <= lim


def _structural_balance(cnt0, cnt1, cost0, cost1, lim) -> np.ndarray:
    """Balance test on the minimum-cost template, used only for error codes.

    Mirrors the forced/flexible split the enumeration assumes: a read is
    forced to its cheapest side (ties prefer fewer mismatches, then side 0);
    an equal-cost read is flexible when its other side also stays within the
    mismatch limit (symmetric ties need the common count within budget,
    asymmetric ties the larger count).  A read whose cheapest side itself
    violates the budget stays forced here, so a purely budget-driven failure
    is not misreported as a balance failure.  This is deliberately looser
    than true feasibility (which the assignment DP decides exactly).
    """
    m = cnt0.shape[0]
    equal = cost0 == cost1
    sym = equal & (cnt0 == cnt1)
    asym = equal & ~sym
    # Cheapest side; an equal-count tie is forced to side 1 (matches the
    # enumeration's canonical cheap-side choice when the tie cannot flex).
    grp0 = (cost0 < cost1) | (equal & (cnt0 < cnt1))
    flex = (sym & (cnt0 <= lim)) | (asym & (np.maximum(cnt0, cnt1) <= lim))
    n0_forced = (grp0 & ~flex).sum(axis=0)
    k_flex = flex.sum(axis=0)
    x_lo = np.maximum(0, MIN_GROUP_SIZE - n0_forced)
    x_hi = np.minimum(k_flex, m - MIN_GROUP_SIZE - n0_forced)
    return x_lo <= x_hi


def _dp_layer(C, T, i, cnt0, cost0, cnt1, cost1, f0, f1):
    """One forward DP layer: decide read i, growing group-0 count by 0 or 1.

    C/T are fixed-size ((m+1), B); rows beyond i read count hold sentinels.
    Each cell stores the lexicographically best (total cost, max per-read
    mismatch count) over prefix assignments.
    """
    m1, B = C.shape
    inf, inf_t = _INF, _INF_T
    src = slice(0, i + 1)
    allow0 = f0[i][None, :]
    allow1 = f1[i][None, :]
    reachable = C[src] < inf

    nc = np.full_like(C, inf)
    nt = np.full_like(T, inf_t)

    # read i -> group 0 shifts the group-0 count from n0 to n0 + 1.
    v0c = np.where(allow0 & reachable, C[src] + cost0[i][None, :], inf)
    v0t = np.where(allow0, np.maximum(T[src], cnt0[i][None, :]), inf_t)
    nc[1:i + 2] = v0c
    nt[1:i + 2] = v0t

    # read i -> group 1 keeps n0; row 0 has no group-0 predecessor conflict.
    v1c = np.where(allow1 & reachable, C[src] + cost1[i][None, :], inf)
    v1t = np.where(allow1, np.maximum(T[src], cnt1[i][None, :]), inf_t)
    if i == 0:
        nc[0] = v1c[0]
        nt[0] = v1t[0]
    else:
        inner = slice(1, i + 1)
        pick0 = (v0c[:-1] < v1c[inner]) | \
                ((v0c[:-1] == v1c[inner]) & (v0c[:-1] != inf)
                 & (v0t[:-1] < v1t[inner]))
        nc[inner] = np.where(pick0, v0c[:-1], v1c[inner])
        nt[inner] = np.where(pick0, v0t[:-1], v1t[inner])
        nc[0] = v1c[0]
        nt[0] = v1t[0]
    return nc, nt


def _scan_candidates(n, obs, mask, costs, limits, lengths):
    """First pass: feasibility and lexicographic optimum per candidate pair.

    Returns
      feasible  : a balanced assignment exists with every read in budget
      totals    : minimum total mismatch cost among balanced assignments
      t_best    : secondary objective (min max per-read mismatch count)
      size_ok   : structural cheap-side balance check (for error codes only;
                  deliberately independent of strict budget feasibility)
      budget_ok : every read has at least one budget-feasible side
    """
    m = obs.shape[0]
    P = 1 << (n - 2)
    feasible = np.zeros(P, dtype=bool)
    totals = np.full(P, _INF, dtype=np.int64)
    t_best = np.full(P, _INF_T, dtype=np.int64)
    size_ok_all = np.zeros(P, dtype=bool)
    budget_ok_all = np.zeros(P, dtype=bool)

    lo, hi = MIN_GROUP_SIZE, m - MIN_GROUP_SIZE
    lim = limits[:, None]
    for start in range(0, P, CHUNK):
        g = np.arange(start, min(start + CHUNK, P), dtype=np.int64)
        cnt0, cost0, cnt1, cost1 = _candidate_columns(
            g, obs, mask, costs, lengths)
        f0, f1 = _feasible_sides(cnt0, cnt1, lim)

        budget_ok = (f0 | f1).all(axis=0)
        # Ungated structural view, matching the original error semantics;
        # true feasibility comes from the DP (best balanced-state cost).
        size_ok = _structural_balance(cnt0, cnt1, cost0, cost1, lim)

        B = g.shape[0]
        C = np.full((m + 1, B), _INF, dtype=np.int64)
        T = np.full((m + 1, B), _INF_T, dtype=np.int64)
        C[0] = 0
        T[0] = 0
        for i in range(m):
            C, T = _dp_layer(C, T, i, cnt0, cost0, cnt1, cost1, f0, f1)

        Cb, Tb = C[lo:hi + 1], T[lo:hi + 1]
        best = Cb.min(axis=0)
        # The DP enforces both per-read budgets and group minimum sizes, so a
        # finite balanced-state cost is the exact feasibility test (including
        # assignments that must pay a premium to keep both groups populated).
        ok = best < _INF
        with np.errstate(invalid="ignore"):
            tmin = np.where(Cb == best, Tb, _INF_T).min(axis=0)

        feasible[g] = ok
        totals[g] = np.where(ok, best, _INF)
        t_best[g] = np.where(ok, tmin, _INF_T)
        size_ok_all[g] = size_ok
        budget_ok_all[g] = budget_ok

    return feasible, totals, t_best, size_ok_all, budget_ok_all


def _reverse_bits(g: np.ndarray, n: int) -> np.ndarray:
    """Numeric order then matches haplotype-string (locus 0 first) order."""
    rev = np.zeros_like(g)
    for j in range(n):
        rev |= ((g >> j) & 1) << (n - 1 - j)
    return rev


# ---------------------------------------------------------------------------
# Exact optimum-assignment bookkeeping for one candidate pair (scalar DP)
# ---------------------------------------------------------------------------

class _AssignmentOptimizer:
    """All optimal assignments of reads to a fixed complementary pair.

    Optimal means minimum total cost, then minimum maximum per-read mismatch
    count, subject to per-read budgets and both groups holding at least
    MIN_GROUP_SIZE reads.  Counts of optimal assignments are saturated at 2
    (only uniqueness and the first two lexicographic assignments are needed),
    which keeps the bookkeeping linear in the (read, group-size) table.
    """

    def __init__(self, sides, limits, m: int, total: int, t_star: int):
        # sides[i] = (cnt0, cost0, cnt1, cost1); feasibility via limits.
        self.sides = sides
        self.limits = limits
        self.m = m
        self.lo = MIN_GROUP_SIZE
        self.hi = m - MIN_GROUP_SIZE
        self.total = total
        self.t_star = t_star

        self.f0 = [sides[i][0] <= limits[i] for i in range(m)]
        self.f1 = [sides[i][2] <= limits[i] for i in range(m)]

        self._build_forward()
        self._build_backward()

    # -- forward / backward tables ----------------------------------------

    def _better(self, a, b) -> bool:
        """True when (cost, max mismatch count) pair a is no worse than b."""
        return a[0] < b[0] or (a[0] == b[0] and a[1] <= b[1])

    def _build_forward(self):
        m, lo, hi = self.m, self.lo, self.hi
        inf = (_INF, _INF_T)
        # fC[i][n0] / fT[i][n0]: best pair on reads 0..i-1 with n0 in group 0
        self.fC = [[_INF] * (m + 1) for _ in range(m + 1)]
        self.fT = [[_INF_T] * (m + 1) for _ in range(m + 1)]
        self.fC[0][0], self.fT[0][0] = 0, 0
        for i in range(m):
            c0, w0, c1, w1 = self.sides[i]
            rowC, rowT = self.fC[i], self.fT[i]
            nxtC, nxtT = self.fC[i + 1], self.fT[i + 1]
            for n0 in range(i + 1):
                if rowC[n0] == _INF:
                    continue
                if self.f0[i]:
                    pair = (rowC[n0] + w0, max(rowT[n0], c0))
                    if self._better(pair, (nxtC[n0 + 1], nxtT[n0 + 1])):
                        nxtC[n0 + 1], nxtT[n0 + 1] = pair
                if self.f1[i]:
                    pair = (rowC[n0] + w1, max(rowT[n0], c1))
                    if self._better(pair, (nxtC[n0], nxtT[n0])):
                        nxtC[n0], nxtT[n0] = pair

    def _build_backward(self):
        m = self.m
        # bC[i][q] / bT[i][q]: best (cost, max mismatch) pair on reads
        # i..m-1 when exactly q of them go to group 0 (balance not applied).
        self.bC = [[_INF] * (m + 1) for _ in range(m + 1)]
        self.bT = [[_INF_T] * (m + 1) for _ in range(m + 1)]
        self.bC[m][0], self.bT[m][0] = 0, 0
        for i in range(m - 1, -1, -1):
            c0, w0, c1, w1 = self.sides[i]
            for q in range(m - i):
                base = self.bC[i + 1][q]
                if base == _INF:
                    continue
                bt = self.bT[i + 1][q]
                if self.f0[i]:
                    pair = (base + w0, max(bt, c0))
                    if self._better(pair, (self.bC[i][q + 1], self.bT[i][q + 1])):
                        self.bC[i][q + 1], self.bT[i][q + 1] = pair
                if self.f1[i]:
                    pair = (base + w1, max(bt, c1))
                    if self._better(pair, (self.bC[i][q], self.bT[i][q])):
                        self.bC[i][q], self.bT[i][q] = pair

        # Second pass: count (saturated at 2) suffix assignments attaining the
        # cell's lexicographic optimum (bC[i][q], bT[i][q]) exactly.  Any such
        # suffix extends some lexicographically optimal prefix, so it never
        # overcounts dead ends.
        self.h = [[0] * (m + 1) for _ in range(m + 1)]
        self.h[m][0] = 1
        for i in range(m - 1, -1, -1):
            c0, w0, c1, w1 = self.sides[i]
            for q in range(m - i):           # group-0 reads after read i
                ways = self.h[i + 1][q]
                if ways == 0:
                    continue
                base, bt = self.bC[i + 1][q], self.bT[i + 1][q]
                if base == _INF:
                    continue
                if (self.f0[i]
                        and base + w0 == self.bC[i][q + 1]
                        and max(bt, c0) == self.bT[i][q + 1]):
                    self.h[i][q + 1] = min(2, self.h[i][q + 1] + ways)
                if (self.f1[i]
                        and base + w1 == self.bC[i][q]
                        and max(bt, c1) == self.bT[i][q]):
                    self.h[i][q] = min(2, self.h[i][q] + ways)

    # -- optimum queries ----------------------------------------------------

    def count_optimal(self) -> int:
        """Saturated (at 2) number of optimal balanced assignments."""
        total = 0
        for n0 in range(self.lo, self.hi + 1):
            if self.fC[self.m][n0] == self.total and self.fT[self.m][n0] == self.t_star:
                total = min(2, total + self._prefix_ways(n0))
        return total

    def _prefix_ways(self, end_n0: int) -> int:
        """Saturated paths to (m, end_n0) attaining that cell's optimum."""
        g = [[0] * (self.m + 1) for _ in range(self.m + 1)]
        g[0][0] = 1
        for i in range(self.m):
            c0, w0, c1, w1 = self.sides[i]
            for n0 in range(i + 1):
                ways = g[i][n0]
                if ways == 0 or self.fC[i][n0] == _INF:
                    continue
                if (self.f0[i]
                        and self.fC[i][n0] + w0 == self.fC[i + 1][n0 + 1]
                        and max(self.fT[i][n0], c0) == self.fT[i + 1][n0 + 1]):
                    g[i + 1][n0 + 1] = min(2, g[i + 1][n0 + 1] + ways)
                if (self.f1[i]
                        and self.fC[i][n0] + w1 == self.fC[i + 1][n0]
                        and max(self.fT[i][n0], c1) == self.fT[i + 1][n0]):
                    g[i + 1][n0] = min(2, g[i + 1][n0] + ways)
        return g[self.m][end_n0]

    def kth_assignment(self, k: int) -> tuple[int, ...]:
        """k-th (1-based) lexicographically smallest optimal assignment."""
        m, lo, hi = self.m, self.lo, self.hi
        assign = [-1] * m
        n0 = 0
        cost_so_far = 0
        t_so_far = 0
        for i in range(m):
            c0, w0, c1, w1 = self.sides[i]
            w0_ways = 0
            if self.f0[i]:
                new_t = max(t_so_far, c0)
                rem_lo = max(0, lo - (n0 + 1))
                rem_hi = min(m - i - 1, hi - (n0 + 1))
                if new_t <= self.t_star:
                    for q in range(rem_lo, rem_hi + 1):
                        if (cost_so_far + w0 + self.bC[i + 1][q] == self.total
                                and max(new_t, self.bT[i + 1][q]) == self.t_star):
                            w0_ways = min(2, w0_ways + self.h[i + 1][q])
            if k <= w0_ways:
                assign[i] = 0
                n0 += 1
                cost_so_far += w0
                t_so_far = max(t_so_far, c0)
            else:
                k -= w0_ways
                assign[i] = 1
                cost_so_far += w1
                t_so_far = max(t_so_far, c1)
        return tuple(assign)


def _candidate_sides(g_index: int, n: int, obs, mask, costs, limits, lengths):
    """Per-read scalar side data for one candidate (used for reconstruction)."""
    g = np.array([g_index], dtype=np.int64)
    cnt0, cost0, cnt1, cost1 = _candidate_columns(g, obs, mask, costs, lengths)
    m = obs.shape[0]
    sides = [(int(cnt0[i, 0]), int(cost0[i, 0]),
              int(cnt1[i, 0]), int(cost1[i, 0])) for i in range(m)]
    return sides, [int(v) for v in limits]


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

    feasible, totals, t_best, size_ok, budget_ok = _scan_candidates(
        n, obs, mask, costs, limits, lengths)

    if not feasible.any():
        code, message = _no_solution_reason(size_ok, budget_ok)
        raise PhaseError(code, message)

    best_total = int(totals[feasible].min())
    # Second objective: minimal maximum per-read mismatch count.
    t_min = int(t_best[feasible & (totals == best_total)].min())

    pool = np.nonzero(feasible & (totals == best_total)
                      & (t_best == t_min))[0]
    # Tie-break on haplotype string (locus 0 first), then assignment tuple.
    pool = pool[np.argsort(_reverse_bits(pool, n), kind="stable")]

    def optimizer_for(g_index: int) -> _AssignmentOptimizer:
        sides, lims = _candidate_sides(
            g_index, n, obs, mask, costs, limits, lengths)
        return _AssignmentOptimizer(sides, lims, m, best_total, t_min)

    g_a = int(pool[0])
    opt_a = optimizer_for(g_a)
    assign_a1 = opt_a.kth_assignment(1)

    second = None
    if opt_a.count_optimal() >= 2:
        second = (g_a, opt_a.kth_assignment(2))
    elif pool.shape[0] > 1:
        g_b = int(pool[1])
        second = (g_b, optimizer_for(g_b).kth_assignment(1))

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
