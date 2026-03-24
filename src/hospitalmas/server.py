"""
HospitalMAS — Web-based diagnostic interface.

Run with:
    uvicorn hospitalmas.server:app --reload --port 8000

Then open http://localhost:8000 in your browser.
"""

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from hospitalmas.main import run_diagnostic_pipeline, _build_runtime_log_file
from hospitalmas.answer_collector import AsyncAnswerCollector

app = FastAPI(title="HospitalMAS Diagnostic API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session store (swap for Redis in production) ────────────────────
_sessions: dict[str, dict[str, Any]] = {}


class DiagnoseRequest(BaseModel):
    user_message: str


class AnswerRequest(BaseModel):
    answer: str


# ── Serve the frontend ────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = Path(__file__).parent / "frontend.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


# ── POST /api/diagnose — start a session ──────────────────────────────────────

@app.post("/api/diagnose")
async def diagnose(req: DiagnoseRequest):
    session_id = str(uuid.uuid4())
    collector = AsyncAnswerCollector()

    _sessions[session_id] = {
        "collector": collector,
        "task": None,
        "result": None,
        "error": None,
        "initial_ranking": None,
    }

    async def _run():
        try:
            result = await run_diagnostic_pipeline(
                req.user_message,
                collector,
                log_file=_build_runtime_log_file(),
            )
            _sessions[session_id]["result"] = result
        except Exception as e:
            _sessions[session_id]["error"] = str(e)

    _sessions[session_id]["task"] = asyncio.create_task(_run())
    return {"session_id": session_id}


# ── GET /api/questions/{id} — SSE stream of questions ─────────────────────────

@app.get("/api/questions/{session_id}")
async def stream_questions(session_id: str):
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    collector: AsyncAnswerCollector = session["collector"]

    async def event_generator():
        while True:
            if session["error"] is not None:
                yield f"data: {json.dumps({'error': session['error']})}\n\n"
                return

            if session["result"] is not None:
                yield f"data: {json.dumps({'done': True, 'result': session['result']})}\n\n"
                return

            try:
                question = await asyncio.wait_for(
                    collector.pending_questions.get(), timeout=2.0
                )
            except asyncio.TimeoutError:
                yield ": heartbeat\n\n"
                continue

            if question.get("done"):
                # No follow-up needed — wait for final result
                while session["result"] is None and session["error"] is None:
                    await asyncio.sleep(0.5)
                if session["error"]:
                    yield f"data: {json.dumps({'error': session['error']})}\n\n"
                else:
                    yield f"data: {json.dumps({'done': True, 'result': session['result']})}\n\n"
                return

            payload = dict(question)
            if question.get("index") == 1 and collector.investigations:
                payload["investigations"] = collector.investigations

            yield f"data: {json.dumps(payload)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── POST /api/answer/{id} — submit one answer ────────────────────────────────

@app.post("/api/answer/{session_id}")
async def submit_answer(session_id: str, req: AnswerRequest):
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    await session["collector"].submit_answer(req.answer)
    return {"status": "received"}


# ── GET /api/result/{id} — poll result ────────────────────────────────────────

@app.get("/api/result/{session_id}")
async def get_result(session_id: str):
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    if session["error"]:
        raise HTTPException(500, session["error"])

    if session["result"] is None:
        return {"status": "processing"}

    return {"status": "complete", "diagnosis": session["result"]}