from contextlib import asynccontextmanager
from fastapi import FastAPI
import httpx

from app.core.config import settings
# from app.core.logging import setup_logging
# from app.core.middleware import RequestIdMiddleware
from app.api.router import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # setup_logging()

    app.state.http = httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT_SECONDS)
    yield
    await app.state.http.aclose()


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        debug=settings.DEBUG,
        lifespan=lifespan,
    )

    # app.add_middleware(RequestIdMiddleware)

    app.include_router(api_router, prefix=settings.API_V1_PREFIX)
    return app


app = create_app()
