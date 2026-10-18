from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.logging_config import RequestLoggingMiddleware, configure_logging
from src.api.routers.assistant import router as assistant_router
from src.api.routers.data_quality import router as dq_router
from src.api.routers.health import router as health_router
from src.api.routers.pipeline import router as pipeline_router
from src.api.routers.port_events import router as port_events_router
from src.api.routers.ports import router as ports_router
from src.api.routers.predict import router as predict_router
from src.api.routers.routes import router as routes_router
from src.api.routers.shipments import router as shipments_router
from src.ml.predict import load_model

app = FastAPI(title="Supply Chain Platform", version="0.1.0")
configure_logging()
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|0\.0\.0\.0)(:\d+)?",
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(shipments_router)
app.include_router(port_events_router)
app.include_router(ports_router)
app.include_router(routes_router)
app.include_router(predict_router)
app.include_router(assistant_router)
app.include_router(dq_router)
app.include_router(pipeline_router)


@app.on_event("startup")
def startup_event() -> None:
    app.state.delay_model = load_model()


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Supply Chain Platform is running."}
