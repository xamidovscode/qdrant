from pydantic import BaseModel


class QuestionResponse(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str

