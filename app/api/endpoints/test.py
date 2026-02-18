from fastapi import APIRouter
from app.api.schemas.test import (
    QuestionResponse,
    AnswerResponse
)

router = APIRouter()

@router.post("/question/", response_model=AnswerResponse)
async def test(data: QuestionResponse):
    return {
        "answer": 'dnx',
        'status': "ok"
    }
