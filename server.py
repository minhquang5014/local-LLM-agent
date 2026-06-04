"""
FastAPI web server — serves the chat UI and streams agent responses via SSE.

Run:
    source venv/bin/activate
    python server.py          # → http://127.0.0.1:8088
"""

import asyncio
import json
import logging
import sys
import threading
from pathlib import Path
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Local LLM Agent", docs_url=None, redoc_url=None)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ------------------------------------------------------------------
# Serve static files (after routes so /api/* takes priority)
# ------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    return (Path(__file__).parent / "static" / "index.html").read_text()


# ------------------------------------------------------------------
# Chat — SSE streaming
# ------------------------------------------------------------------

# Per-session conversation history:  session_id -> [(user_msg, assistant_msg), ...]
session_histories: dict[str, list] = {}


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _run():
        try:
            from src.agent import stream_agent
            history = list(session_histories.get(req.session_id, []))
            assistant_answer = ""
            for event in stream_agent(req.message, history):
                asyncio.run_coroutine_threadsafe(queue.put(event), loop)
                if event.get("type") == "answer":
                    assistant_answer = event.get("content", "")
            if assistant_answer:
                history.append((req.message, assistant_answer))
                session_histories[req.session_id] = history
        except Exception as e:
            asyncio.run_coroutine_threadsafe(
                queue.put({"type": "error", "message": str(e)}), loop
            )
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    threading.Thread(target=_run, daemon=True).start()

    async def event_stream() -> AsyncGenerator[str, None]:
        while True:
            event = await queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ------------------------------------------------------------------
# Memory
# ------------------------------------------------------------------

@app.get("/api/memory")
async def get_memory():
    try:
        from src.memory import MemoryStore
        mem = MemoryStore()
        raw = mem._collection.get(include=["documents", "metadatas"])
        items = [
            {"id": id_, "text": doc, "source": meta.get("source", "")}
            for id_, doc, meta in zip(raw["ids"], raw["documents"], raw["metadatas"])
        ]
        return {"items": items[-50:], "count": len(items)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/memory")
async def clear_memory():
    from src.memory import MemoryStore
    MemoryStore().clear()
    return {"status": "cleared"}


# ------------------------------------------------------------------
# Health
# ------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {"status": "ok", "model": "Qwen3.5-9B (OptiQ · mlx-lm)"}


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    import socket
    try:
        # Connect a UDP socket to an external address to find the real outbound IP.
        # No data is sent — this just makes the OS pick the right network interface.
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "run 'ifconfig | grep inet' on Mac to find your LAN IP"
    print("\n  Local LLM Agent")
    print("  ─────────────────────────────")
    print(f"  Local  → http://127.0.0.1:8088")
    print(f"  Network→ http://{local_ip}:8088  (other devices on same WiFi)")
    print()
    uvicorn.run(app, host="0.0.0.0", port=8088, log_level="warning")
