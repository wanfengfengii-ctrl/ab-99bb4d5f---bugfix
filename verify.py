"""One-shot verification for the ancient-DNA phasing service.

Runs after the API is healthy and aggregates three kinds of checks into a
single process exit code:

  * service reachability / readiness  (GET /health)
  * API smoke checks against POST /api/phase, including a sample that forces
    mismatches, an ambiguous sample with two distinct optimal solutions, a
    non-contiguous read (business error) and an infeasible budget;
  * the project's unit-test suite (pytest).

Exit code bits (0 = everything passed):
    1  at least one API smoke check failed
    2  the unit-test suite failed
    4  the service never reported healthy

Usage:
    python verify.py [base_url]
Environment:
    BASE_URL  default http://localhost:8000 (CLI argument wins)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

HEALTH_BIT = 4
SMOKE_BIT = 1
UNIT_BIT = 2
HEALTH_TIMEOUT_S = 60


# ---------------------------------------------------------------------------
# Deterministic fixtures
# ---------------------------------------------------------------------------

def _read(positions, obs, limit=None, costs=None):
    return {
        "positions": positions,
        "observations": obs,
        "mismatch_costs": costs if costs is not None else [1] * len(positions),
        "max_mismatches": len(positions) if limit is None else limit,
    }


MISMATCH_SAMPLE = {
    # True pair is 00000000 / 11111111; read 0 contains one degraded base at
    # locus 1.  Overlapping reads span the whole interval, pinning the phase.
    "loci": 8,
    "reads": [
        _read([0, 1, 2], "010", costs=[4, 1, 4]),
        _read([0, 1, 2, 3], "0000"),
        _read([2, 3, 4, 5], "0000"),
        _read([3, 4, 5, 6, 7], "00000"),
        _read([0, 1], "00"),
        _read([0, 1, 2, 3], "1111"),
        _read([2, 3, 4, 5], "1111"),
        _read([3, 4, 5, 6, 7], "11111"),
        _read([6, 7], "11"),
        _read([4, 5], "11"),
    ],
}

AMBIGUOUS_SAMPLE = {
    # 00000000/11111111 and 11110000/00001111 both explain every read with
    # zero mismatches -> genuinely ambiguous optimum.
    "loci": 8,
    "reads": [
        _read([0, 1, 2, 3], "0000"),
        _read([0, 1, 2], "000"),
        _read([0, 1, 2, 3], "1111"),
        _read([1, 2, 3], "111"),
        _read([4, 5, 6, 7], "0000"),
        _read([4, 5, 6], "000"),
        _read([4, 5, 6, 7], "1111"),
        _read([5, 6, 7], "111"),
        _read([0, 1], "00"),
        _read([4, 5], "00"),
    ],
}


# Eight zero-mismatch anchor reads (four perfect 0-reads and four perfect
# 1-reads, all with max_mismatches=0) cover loci 0..7 and pin the canonical
# pair uniquely to 00000000 / 11111111.  The ninth read X covers loci 0..2 and
# exercises each side-feasibility category; the tenth read is a zero-budget
# 0-read on 4..5.
_ANCHOR_READS = [
    _read([0, 1, 2], "000", limit=0),
    _read([3, 4, 5, 6], "0000", limit=0),
    _read([6, 7], "00", limit=0),
    _read([2, 3, 4], "000", limit=0),
    _read([0, 1, 2], "111", limit=0),
    _read([3, 4, 5, 6], "1111", limit=0),
    _read([5, 6, 7], "111", limit=0),
    _read([0, 1], "11", limit=0),
]


def _budget_sample(x_read):
    return {
        "loci": 8,
        "reads": _ANCHOR_READS + [x_read, _read([4, 5], "00", limit=0)],
    }


# Regression for the one-side-exceeds-budget bug: X = 110 with costs
# [1,1,100] and a one-mismatch budget.  Group 0 (000) needs two cheap
# mismatches (cost 2 but illegal); group 1 (111) needs one costly mismatch at
# locus 2 (cost 100, legal) -- the valid global optimum used to be rejected
# with MISMATCH_BUDGET_EXCEEDED.
ONE_SIDED_BUDGET_SAMPLE = _budget_sample(
    _read([0, 1, 2], "110", costs=[1, 1, 100], limit=1))

# Both sides are within X's budget.  Group 0 costs 5 with one mismatch, group 1
# costs 2 with two mismatches: total cost wins even at the larger count.
BOTH_SIDES_SAMPLE = _budget_sample(
    _read([0, 1, 2], "010", costs=[1, 5, 1], limit=3))

# Equal cost (4) but one mismatch on group 0 vs two on group 1: the second
# objective (maximum per-read mismatch count) breaks the tie.
EQUAL_COST_SAMPLE = _budget_sample(
    _read([0, 1, 2], "010", costs=[3, 4, 1], limit=2))

# Neither side fits a zero-mismatch budget: the budget error must survive.
NEITHER_SIDE_SAMPLE = _budget_sample(
    _read([0, 1, 2], "010", limit=0))


def _gapped_sample():
    payload = json.loads(json.dumps(MISMATCH_SAMPLE))
    payload["reads"][0] = _read([0, 2], "00")
    return payload


def _infeasible_sample():
    reads = [_read([0, 1], s, limit=0) for s in ("00", "01", "10", "11")]
    reads += [_read([p], str(p % 2), limit=0) for p in range(2, 8)]
    return {"loci": 8, "reads": reads}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _request(base_url, path, payload=None):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(base_url.rstrip("/") + path, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


def _check(name, ok, detail=""):
    marker = "PASS" if ok else "FAIL"
    print(f"  [{marker}] {name}" + (f" -- {detail}" if detail and not ok else ""))
    return ok


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def wait_for_health(base_url):
    deadline = time.time() + HEALTH_TIMEOUT_S
    while time.time() < deadline:
        try:
            status, body = _request(base_url, "/health")
            if status == 200 and body.get("status") == "ok":
                print(f"  service healthy at {base_url}")
                return True
        except Exception:
            pass
        time.sleep(1)
    print(f"  [FAIL] service did not become healthy within {HEALTH_TIMEOUT_S}s")
    return False


def run_smoke(base_url):
    results = []

    print("- health endpoint")
    status, body = _request(base_url, "/health")
    results.append(_check("GET /health returns 200 ok", status == 200 and body == {"status": "ok"},
                          f"got {status} {body}"))

    print("- sample containing degraded mismatches")
    status, body = _request(base_url, "/api/phase", MISMATCH_SAMPLE)
    ok = status == 200
    if ok:
        sol = body["solutions"][0]
        mm = [a for a in sol["assignments"] if a["mismatch_count"] > 0]
        ok = (
            body["status"] == "unique"
            and body["objective"]["total_mismatch_cost"] == 1
            and body["objective"]["max_mismatches_per_read"] == 1
            and len(mm) == 1 and mm[0]["read_id"] == 0
            and mm[0]["mismatches"][0]["position"] == 1
            and mm[0]["mismatches"][0]["cost"] == 1
            and all(s >= 2 for s in sol["group_sizes"])
        )
    results.append(_check("unique optimum, cost 1, one mismatch at locus 1",
                          ok, f"status={status} body={body if status != 200 else body.get('objective')}"))

    print("- expensive side is the only within-budget choice")
    status, body = _request(base_url, "/api/phase", ONE_SIDED_BUDGET_SAMPLE)
    ok = False
    if status == 200:
        sol = body["solutions"][0]
        x = sol["assignments"][8]
        ok = (
            body["status"] == "unique"
            and body["objective"]["total_mismatch_cost"] == 100
            and body["objective"]["max_mismatches_per_read"] == 1
            and sol["haplotypes"] == {"group_0": "00000000",
                                      "group_1": "11111111"}
            and [a["group"] for a in sol["assignments"]] ==
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 0]
            and sol["group_sizes"] == [5, 5]
            and x["group"] == 1 and x["mismatch_count"] == 1
            and x["mismatch_cost"] == 100
            and x["mismatches"] == [
                {"position": 2, "observed": "0", "expected": "1", "cost": 100}]
        )
    results.append(_check("unique optimum via costly side, cost 100 at locus 2",
                          ok, f"status={status} body={body if status != 200 else body.get('objective')}"))

    print("- both sides feasible: total cost decides first")
    status, body = _request(base_url, "/api/phase", BOTH_SIDES_SAMPLE)
    ok = False
    if status == 200:
        x = body["solutions"][0]["assignments"][8]
        ok = (
            body["status"] == "unique"
            and body["objective"] == {"total_mismatch_cost": 2,
                                      "max_mismatches_per_read": 2}
            and x["group"] == 1 and x["mismatch_count"] == 2
            and x["mismatch_cost"] == 2
        )
    results.append(_check("cheaper side (cost 2, 2 mismatches) wins over cost 5, 1",
                          ok, f"status={status}"))

    print("- equal cost, different mismatch counts: threshold breaks the tie")
    status, body = _request(base_url, "/api/phase", EQUAL_COST_SAMPLE)
    ok = False
    if status == 200:
        x = body["solutions"][0]["assignments"][8]
        ok = (
            body["status"] == "unique"
            and body["objective"] == {"total_mismatch_cost": 4,
                                      "max_mismatches_per_read": 1}
            and x["group"] == 0 and x["mismatch_count"] == 1
            and x["mismatch_cost"] == 4
        )
    results.append(_check("equal cost 4 resolved to one mismatch",
                          ok, f"status={status}"))

    print("- neither side within budget keeps the budget error")
    status, body = _request(base_url, "/api/phase", NEITHER_SIDE_SAMPLE)
    results.append(_check("422 MISMATCH_BUDGET_EXCEEDED (neither side feasible)",
                          status == 422
                          and body.get("error", {}).get("code") == "MISMATCH_BUDGET_EXCEEDED",
                          f"got {status} {body}"))

    print("- ambiguous sample (two distinct optima)")
    status, body = _request(base_url, "/api/phase", AMBIGUOUS_SAMPLE)
    ok = False
    if status == 200 and body["status"] == "ambiguous" and len(body["solutions"]) == 2:
        s1, s2 = body["solutions"]
        distinct = (s1["haplotypes"] != s2["haplotypes"]) or \
                   ([a["group"] for a in s1["assignments"]] !=
                    [a["group"] for a in s2["assignments"]])
        same_objective = (s1["total_mismatch_cost"] == s2["total_mismatch_cost"]
                          and s1["max_mismatches_per_read"] == s2["max_mismatches_per_read"])
        complementary = all(
            s["haplotypes"]["group_1"] == "".join(
                "1" if c == "0" else "0" for c in s["haplotypes"]["group_0"])
            for s in (s1, s2)
        )
        ok = distinct and same_objective and complementary
    results.append(_check("ambiguous, two distinct tied solutions returned",
                          ok, f"status={status}"))

    print("- non-contiguous read rejected with a business reason")
    status, body = _request(base_url, "/api/phase", _gapped_sample())
    results.append(_check("422 READ_NOT_CONTIGUOUS",
                          status == 422 and body.get("error", {}).get("code") == "READ_NOT_CONTIGUOUS",
                          f"got {status} {body}"))

    print("- infeasible mismatch budgets rejected with a business reason")
    status, body = _request(base_url, "/api/phase", _infeasible_sample())
    results.append(_check("422 MISMATCH_BUDGET_EXCEEDED",
                          status == 422 and body.get("error", {}).get("code") in
                          ("MISMATCH_BUDGET_EXCEEDED", "INFEASIBLE_GROUP_BALANCE"),
                          f"got {status} {body}"))

    return all(results)


def run_unit_tests():
    root = os.path.dirname(os.path.abspath(__file__))
    print("- unit tests (pytest)")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q"],
        cwd=root,
    )
    return proc.returncode == 0


def main(argv):
    base_url = argv[1] if len(argv) > 1 else os.environ.get("BASE_URL", "http://localhost:8000")
    print(f"== Verifying phasing API at {base_url} ==")

    print("[1/3] waiting for service health")
    healthy = wait_for_health(base_url)

    smoke_ok = False
    unit_ok = False
    if healthy:
        print("[2/3] API smoke checks")
        smoke_ok = run_smoke(base_url)
        print("[3/3] code tests")
        unit_ok = run_unit_tests()

    code = 0
    if not healthy:
        code |= HEALTH_BIT
    if healthy and not smoke_ok:
        code |= SMOKE_BIT
    if healthy and not unit_ok:
        code |= UNIT_BIT

    print()
    if code == 0:
        print("VERIFY RESULT: PASS (health, API smoke, and unit tests all passed)")
    else:
        parts = []
        if code & HEALTH_BIT:
            parts.append("service-not-healthy")
        if code & SMOKE_BIT:
            parts.append("api-smoke-failed")
        if code & UNIT_BIT:
            parts.append("unit-tests-failed")
        print(f"VERIFY RESULT: FAIL (exit {code}: {', '.join(parts)})")
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv))
