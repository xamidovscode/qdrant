from fastapi import APIRouter
from app.api.endpoints import health, test

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(test.router, tags=["test"])
