from fastapi import APIRouter
from app.api.schemas.common import HealthResponse

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def healthcheck():
    return HealthResponse(status="ok")


