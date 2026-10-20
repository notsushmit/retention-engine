"""The scoring API: GET /health and POST /score (configs/serving.yaml, contract and api).

Startup loads the models through the registry alias with every check; a missing
alias, a missing, extra or changed file, or a manifest that differs refuses
startup (the exception ends the app's lifespan before it serves anything). A
Valkey that cannot be reached does not stop startup: the service starts not
ready and says why.

Every error body is the registered one: 404 unknown customer, 409 as_of differs
from the store's date, 422 malformed input (the validation details are never
returned, so no field name a caller sent comes back), 503 not ready with the
reason. A response that cannot be built from the registered schemas is a 500,
never a partial answer.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from retention.api import schemas as sch
from retention.api import service as svc
from retention.api import store


def error(status_code: int, reason: str | None = None) -> JSONResponse:
    body = sch.error_body(status_code, reason).model_dump(mode="json")
    return JSONResponse(status_code=status_code, content=body)


def create_app(
    load: Callable[[], svc.Context] = svc.default_context,
    connect: Callable[[], object] = store.client,
) -> FastAPI:
    """The app; `load` builds the scoring context at startup, `connect` the Valkey client."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        ctx = load()  # raises on any registry or checksum problem: startup is refused
        app.state.ctx, app.state.valkey = ctx, connect()
        try:
            yield
        finally:
            ctx.close()

    app = FastAPI(title="Retention scoring", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def malformed(request: Request, exc: RequestValidationError) -> JSONResponse:
        return error(422)

    @app.exception_handler(svc.NotReady)
    async def not_ready(request: Request, exc: svc.NotReady) -> JSONResponse:
        return error(503, str(exc))

    @app.exception_handler(svc.DateDiffers)
    async def date_differs(request: Request, exc: svc.DateDiffers) -> JSONResponse:
        return error(409)

    @app.exception_handler(svc.UnknownCustomer)
    async def unknown(request: Request, exc: svc.UnknownCustomer) -> JSONResponse:
        return error(404)

    @app.get("/health", response_model=svc.Health)
    def get_health(request: Request) -> JSONResponse:
        h = svc.health(request.app.state.ctx, request.app.state.valkey)
        return JSONResponse(status_code=200 if h.ready else 503, content=h.model_dump(mode="json"))

    @app.post("/score", response_model=sch.Scored | sch.Outside | sch.NotActive)
    def post_score(req: svc.ScoreRequest, request: Request) -> JSONResponse:
        body = svc.answer(request.app.state.ctx, request.app.state.valkey, req)
        return JSONResponse(status_code=200, content=body)

    return app
