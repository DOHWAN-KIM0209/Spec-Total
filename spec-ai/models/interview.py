# models/interview.py
from sqlmodel import SQLModel, Field

class InterviewQuestion(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    category: str  # 질문 카테고리 (예: 공통, 이력서)
    question: str  # 질문 내용
