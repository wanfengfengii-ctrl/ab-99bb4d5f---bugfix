"""Tests for the phasing solver, including an exhaustive oracle."""

from __future__ import annotations

import itertools
import time

import numpy as np
import pytest

from app.phaser import (
    MIN_GROUP_SIZE,
    PhaseError,
    parse_input,
    solve,
)


# ---------------------------------------------------------------------------
# Exhaustive brute-force oracle (small instances only)
# ---------------------------------------------------------------------------

def _hap_str(g: int, n: int) -> str:
    return "".join(str((g >> j) & 1) for j in range(n))


def brute_force(payload: dict):
    """Enumerate every canonical haplotype pair and every assignment."""
    n = payload["loci"]
    reads = payload["reads"]
    m = len(reads)

    best = None
    best_rows = []
    for g in range(1 << (n - 2)):
        h = _hap_str(g, n)
        hbar = "".join("1" if c == "0" else "0" for c in h)
        read_opts = []
        for r in reads:
            opts = []
            for side in (0, 1):
                seq = h if side == 0 else hbar
                cnt = cost = 0
                for j, p in enumerate(r["positions"]):
                    if r["observations"][j] != seq[p]:
                        cnt += 1
                        cost += r["mismatch_costs"][j]
                opts.append((side, cnt, cost))
            read_opts.append(opts)
        for bits in range(1 << m):
            assignment = tuple((bits >> i) & 1 for i in range(m))
            if (sum(1 for a in assignment if a == 0) < MIN_GROUP_SIZE
                    or sum(1 for a in assignment if a == 1) < MIN_GROUP_SIZE):
                continue
            total = t = 0
            feasible = True
            for i, a in enumerate(assignment):
                _, cnt, cost = read_opts[i][a]
                if cnt > reads[i]["max_mismatches"]:
                    feasible = False
                    break
                total += cost
                t = max(t, cnt)
            if not feasible:
                continue
            key = (total, t, h, assignment)
            score = (total, t)
            if best is None or score < best:
                best = score
                best_rows = [key]
            elif score == best:
                best_rows.append(key)
    if best is None:
        return None
    # every collected row has the optimal (total cost, max mismatch) pair
    return best, best_rows


# ---------------------------------------------------------------------------
# Random instance generation around a "true" haplotype
# ---------------------------------------------------------------------------

def make_payload(rng, n=8, m=10, noise=0.12, max_len=None):
    if max_len is None:
        max_len = n
    g_star = int(rng.integers(0, 1 << n))
    reads = []
    seen = set()
    while len(reads) < m:
        L = int(rng.integers(2, max_len + 1))
        start = int(rng.integers(0, n - L + 1))
        positions = list(range(start, start + L))
        side = int(rng.integers(0, 2))
        obs = []
        for p in positions:
            allele = ((g_star >> p) & 1) ^ side
            if rng.random() < noise:
                allele ^= 1
            obs.append(str(allele))
        obs = "".join(obs)
        if (tuple(positions), obs) in seen:
            continue
        seen.add((tuple(positions), obs))
        reads.append({
            "positions": positions,
            "observations": obs,
            "mismatch_costs": [int(rng.integers(1, 5)) for _ in positions],
            "max_mismatches": L,  # permissive; oracle decides feasibility
        })
    return {"loci": n, "reads": reads}


def _result_assignment(result, rank=1):
    sol = result["solutions"][rank - 1]
    return (sol["haplotypes"]["group_0"],
            tuple(a["group"] for a in sol["assignments"]))


@pytest.mark.parametrize("seed", range(40))
def test_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    payload = make_payload(rng)
    result = solve(payload)
    oracle = brute_force(payload)
    assert oracle is not None, f"seed {seed}: random instance happened to be infeasible"
    best, rows = oracle
    total, t = best
    h, assignment = rows[0][2], rows[0][3]

    assert result["objective"]["total_mismatch_cost"] == total
    assert result["objective"]["max_mismatches_per_read"] == t

    first = min(rows, key=lambda k: (k[2], k[3]))
    h0, a = _result_assignment(result)
    assert h0 == first[2]
    assert a == first[3]

    if len(rows) == 1:
        assert result["status"] == "unique"
        assert len(result["solutions"]) == 1
    else:
        assert result["status"] == "ambiguous"
        assert len(result["solutions"]) == 2
        h0b, ab = _result_assignment(result, 2)
        expected_second = min(rows[1:], key=lambda k: (k[2], k[3]))
        assert (h0b, ab) == (expected_second[2], expected_second[3])


def test_brute_force_on_permissive_budget():
    rng = np.random.default_rng(123)
    payload = make_payload(rng, noise=0.0)
    result = solve(payload)
    (total, _), _ = brute_force(payload)
    assert result["objective"]["total_mismatch_cost"] == total


@pytest.mark.parametrize("seed", range(12))
def test_matches_brute_force_tight_budget(seed):
    # Limit each read to one mismatch: exercises budget-constrained feasibility
    # and infeasibility paths against the exhaustive oracle.
    rng = np.random.default_rng(1000 + seed)
    payload = make_payload(rng, noise=0.08)
    for r in payload["reads"]:
        r["max_mismatches"] = 1
    oracle = brute_force(payload)
    if oracle is None:
        with pytest.raises(PhaseError):
            solve(payload)
        return
    (total, t), rows = oracle
    result = solve(payload)
    assert result["objective"]["total_mismatch_cost"] == total
    assert result["objective"]["max_mismatches_per_read"] == t
    assert result["objective"]["max_mismatches_per_read"] <= 1
    if len(rows) == 1:
        assert result["status"] == "unique"
    else:
        assert result["status"] == "ambiguous"
        assert len(result["solutions"]) == 2


# ---------------------------------------------------------------------------
# Validation and business errors
# ---------------------------------------------------------------------------

def _valid_read(start, length, obs, costs=None, limit=None):
    positions = list(range(start, start + length))
    return {
        "positions": positions,
        "observations": obs,
        "mismatch_costs": costs if costs is not None else [1] * length,
        "max_mismatches": length if limit is None else limit,
    }


def _valid_payload():
    reads = [
        _valid_read(i % 5, 3, format(i % 8, "03b"))
        for i in range(10)
    ]
    return {"loci": 8, "reads": reads}


def test_gapped_read_rejected():
    payload = _valid_payload()
    payload["reads"][3]["positions"] = [0, 2, 3]
    with pytest.raises(PhaseError) as exc:
        solve(payload)
    assert exc.value.code == "READ_NOT_CONTIGUOUS"


def test_unsorted_positions_rejected():
    payload = _valid_payload()
    payload["reads"][0]["positions"] = [2, 1, 0]
    with pytest.raises(PhaseError) as exc:
        solve(payload)
    assert exc.value.code == "VALIDATION_ERROR"


def test_loci_count_bounds():
    with pytest.raises(PhaseError):
        solve({"loci": 7, "reads": []})
    with pytest.raises(PhaseError):
        solve({"loci": 19, "reads": []})


def test_read_count_bounds():
    payload = _valid_payload()
    payload["reads"] = payload["reads"][:9]
    with pytest.raises(PhaseError):
        solve(payload)


def test_nonpositive_cost_rejected():
    payload = _valid_payload()
    payload["reads"][0]["mismatch_costs"][0] = 0
    with pytest.raises(PhaseError) as exc:
        solve(payload)
    assert exc.value.code == "VALIDATION_ERROR"


def test_duplicate_read_rejected():
    payload = _valid_payload()
    payload["reads"][1] = dict(payload["reads"][0])
    with pytest.raises(PhaseError) as exc:
        solve(payload)
    assert exc.value.code == "DUPLICATE_READ"


def test_infeasible_zero_budget():
    # Four distinct observations over the same interval: two homologs can
    # match at most two of them (complementary pair), and each homolog must
    # hold >= 2 reads, so at least one mismatch is unavoidable.
    reads = [
        {"positions": [0, 1], "observations": s,
         "mismatch_costs": [1, 1], "max_mismatches": 0}
        for s in ("00", "01", "10", "11")
    ]
    for p in range(2, 8):
        reads.append({
            "positions": [p], "observations": str(p % 2),
            "mismatch_costs": [1], "max_mismatches": 0,
        })
    with pytest.raises(PhaseError) as exc:
        solve({"loci": 8, "reads": reads})
    assert exc.value.code in {"MISMATCH_BUDGET_EXCEEDED", "INFEASIBLE_GROUP_BALANCE"}


def test_missing_body_fields():
    with pytest.raises(PhaseError):
        parse_input({"loci": 8})


# ---------------------------------------------------------------------------
# Structural properties
# ---------------------------------------------------------------------------

def test_haplotypes_are_complementary_and_canonical():
    rng = np.random.default_rng(7)
    result = solve(make_payload(rng))
    for sol in result["solutions"]:
        a = sol["haplotypes"]["group_0"]
        b = sol["haplotypes"]["group_1"]
        assert len(a) == 8
        assert b == "".join("1" if c == "0" else "0" for c in a)
        assert a[-2:] == "00"  # canonical: two leading loci (n-2, n-1) zero


def test_group_sizes_and_evidence_consistency():
    rng = np.random.default_rng(99)
    payload = make_payload(rng)
    result = solve(payload)
    sol = result["solutions"][0]
    assert all(s >= MIN_GROUP_SIZE for s in sol["group_sizes"])
    assert sum(sol["group_sizes"]) == 10
    total = 0
    for ev, r in zip(sol["assignments"], payload["reads"]):
        assert len(ev["mismatches"]) == ev["mismatch_count"]
        assert ev["mismatch_cost"] == sum(m["cost"] for m in ev["mismatches"])
        assert ev["within_mismatch_limit"]
        for mm in ev["mismatches"]:
            assert mm["observed"] != mm["expected"]
            assert mm["expected"] == sol["haplotypes"][f"group_{ev['group']}"][mm["position"]]
        total += ev["mismatch_cost"]
    assert total == sol["total_mismatch_cost"]


def test_cost_priority_over_max_mismatches():
    """Cheaper total cost must win even with a larger per-read mismatch count."""
    rng = np.random.default_rng(2024)
    payload = make_payload(rng)
    # Inflate one locus cost heavily on several reads so that a 2-mismatch
    # routing can beat a 1-mismatch routing in total cost.
    for r in payload["reads"]:
        r["mismatch_costs"] = [100 if p % 3 == 0 else 1 for p in r["positions"]]
    result = solve(payload)
    (total, t), _ = brute_force(payload)
    assert result["objective"]["total_mismatch_cost"] == total
    assert result["objective"]["max_mismatches_per_read"] == t


def test_deterministic_across_calls():
    rng = np.random.default_rng(55)
    payload = make_payload(rng)
    a = solve(payload)
    b = solve(payload)
    assert a == b


def test_worst_case_runtime():
    rng = np.random.default_rng(31337)
    n, m = 18, 36
    g_star = int(rng.integers(0, 1 << n))
    reads = []
    seen = set()
    while len(reads) < m:
        L = int(rng.integers(6, 18))
        start = int(rng.integers(0, n - L + 1))
        positions = list(range(start, start + L))
        side = int(rng.integers(0, 2))
        obs = "".join(str(((g_star >> p) & 1) ^ side) for p in positions)
        key = (tuple(positions), obs)
        if key in seen:
            continue
        seen.add(key)
        reads.append({
            "positions": positions,
            "observations": obs,
            "mismatch_costs": [int(rng.integers(1, 9)) for _ in positions],
            "max_mismatches": L,
        })
    start = time.perf_counter()
    result = solve({"loci": n, "reads": reads})
    elapsed = time.perf_counter() - start
    assert result["status"] in {"unique", "ambiguous"}
    assert elapsed < 20, f"solve took {elapsed:.1f}s"


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------

def test_http_smoke():
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}

    rng = np.random.default_rng(4242)
    payload = make_payload(rng)
    resp = client.post("/api/phase", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in {"unique", "ambiguous"}
    assert len(body["solutions"][0]["assignments"]) == 10

    bad = {"loci": payload["loci"], "reads": [dict(r) for r in payload["reads"]]}
    bad["reads"][0] = {
        "positions": [0, 2], "observations": "00",
        "mismatch_costs": [1, 1], "max_mismatches": 2,
    }
    resp = client.post("/api/phase", json=bad)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "READ_NOT_CONTIGUOUS"
