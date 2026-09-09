from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    fastapi_app = FastAPI(title=settings.app_name)
    fastapi_app.mount("/static", StaticFiles(directory="app/static"), name="static")
    fastapi_app.mount("/result-artifacts", StaticFiles(directory="results"), name="result_artifacts")
    fastapi_app.include_router(router)
    return fastapi_app


app = create_app()
