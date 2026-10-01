"""HTTP service: POST /api/route takes one request as JSON and returns the team, confidence and reasons.
GET / serves the screen that calls it. Starts without any API key.
"""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import clarify
from .engine import ARTIFACT, ASK_BELOW, Router

STATIC = Path(__file__).resolve().parent / "static"


class ServiceRequest(BaseModel):
    request_text: str = Field(..., min_length=1, max_length=4000, description="Customer's opening message or IVR transcript")
    product_family: Optional[Literal["Water Purifier", "Air Fryer", "Mixer Grinder", "Induction Cooktop", "Room Heater",
                                     "Ceiling Fan", "Robot Vacuum"]] = None
    warranty_status: Optional[Literal["in_warranty", "shield", "out_of_warranty"]] = None
    channel: Optional[Literal["ivr", "chat", "whatsapp", "email"]] = None
    request_id: Optional[str] = None


def _load_router() -> Router:
    import sklearn
    from . import data, train
    try:
        r = Router.load(ARTIFACT)
        if r.meta.get("sklearn") == sklearn.__version__:
            return r
        reason = f"saved with scikit-learn {r.meta.get('sklearn')}, this machine has {sklearn.__version__}"
    except Exception as e:                  # missing or unreadable
        r, reason = None, str(e)
    try:
        return train.train_and_save(data.DATA)  # a few seconds
    except FileNotFoundError:
        if r is not None:                   # no data to retrain from: use the saved model as it is
            return r
        raise RuntimeError(f"cannot load the saved model ({reason}) and data/ has no training files")


router = None
startup_error = None


def startup():
    global router, startup_error
    try:
        router = _load_router()
    except Exception as e:                  # no model and no data to train one from
        startup_error = f"No trained model in models/ and no training data in data/ ({e})."


@asynccontextmanager
async def lifespan(_app):
    startup()
    yield


app = FastAPI(title="Kestrel service-request router", version="1.0", lifespan=lifespan)


@app.get("/api/health")
def health():
    import os
    return {"ready": router is not None, "error": startup_error,
            "model": router.meta if router else None, "ask_below": ASK_BELOW,
            "question_writer": "Groq" if os.environ.get("GROQ_API_KEY") else "standard question (no GROQ_API_KEY)"}


@app.post("/api/route")
def route(req: ServiceRequest):
    if router is None:
        raise HTTPException(503, startup_error or "Model not loaded")
    out = router.route(req.model_dump())
    if out["action"] == "ask_first":
        out["ask_customer"] = clarify.ask(req.request_text, req.product_family or "product",
                                          [out["team"], out["second_choice"]])
    return out


@app.get("/")
def screen():
    return FileResponse(STATIC / "index.html")
