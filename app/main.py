"""HTTP API for the ancient-DNA haplotype phasing service."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .phaser import PhaseError, solve

app = FastAPI(
    title="Ancient DNA Phasing API",
    version="1.0.0",
    description="Jointly reconstruct a complementary haplotype pair and the "
                "unique homolog assignment of every degraded read.",
)


@app.exception_handler(PhaseError)
async def phase_error_handler(_: Request, exc: PhaseError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message}},
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
async def root() -> dict[str, object]:
    return {
        "service": "ancient-dna-phasing",
        "endpoints": {
            "POST /api/phase": "phase reads into complementary haplotypes",
            "GET /health": "liveness/readiness probe",
        },
    }


@app.post("/api/phase")
async def phase(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(
            status_code=422,
            content={"error": {
                "code": "VALIDATION_ERROR",
                "message": "request body must be valid JSON",
            }},
        )
    result = solve(payload)
    return JSONResponse(result)
