"""The ASGI app factory: routers, registry, and error mapping."""

from contextlib import asynccontextmanager

import duckdb
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .. import service
from ..config import Config, ConfigError
from ..query import ArtifactNotFound, RefNotFound, SchemaMismatch
from . import api
from .deps import SourceRegistry

# One place ties each typed failure to a status:
#   * unknown source / unresolvable ref / no rows → 404
#   * malformed input (bad regex, bad cursor)     → 400
#   * artifact missing or built with another schema → 503, the fix being
#     `obohog source sync` — the server is fine, the data needs work.
_STATUS = {
    ConfigError: 404,
    RefNotFound: 404,
    service.InvalidCursor: 400,
    duckdb.InvalidInputException: 400,
    ArtifactNotFound: 503,
    SchemaMismatch: 503,
}


def create_app(cfg: Config) -> FastAPI:
    """Build the app over an explicit Config (no cwd lookups in here)."""
    registry = SourceRegistry(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        registry.close()

    app = FastAPI(
        title="obohog",
        description="Queryable history of OBO ontology term evolution.",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    app.state.config = cfg
    app.state.registry = registry
    app.include_router(api.router, prefix="/api/v1")

    for exc_type, status in _STATUS.items():
        app.add_exception_handler(exc_type, _handler(status))

    return app


def _handler(status: int):
    def handle(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=status)

    return handle
