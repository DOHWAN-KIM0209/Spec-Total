from datetime import datetime
from enum import Enum
from typing import Optional  # ✅ 여기 추가

import sqlmodel
from models.common import CommonResponse
from pydantic import BaseModel, Field
from sqlmodel import SQLModel


class AnalysisRequest(BaseModel):
    """분석 요청 파라미터"""

    analysis_id: int = Field(description="분석 아이디", ge=1)


class AnalysisResponse(CommonResponse):
    """분석 응답"""


class Status(Enum):
    process = "process"
    success = "success"
    fail = "fail"


class Analysis(SQLModel, table=True):
    id: int = sqlmodel.Field(primary_key=True)

    question: str = sqlmodel.Field(max_length=512)
    answer: str = sqlmodel.Field(max_length=1024)

    video_path: str = sqlmodel.Field(max_length=1024)
    thumbnail_path: str = sqlmodel.Field(max_length=1024)

    analysis_start_time: Optional[datetime] = sqlmodel.Field(default=None, nullable=True)
    analysis_end_time: Optional[datetime] = sqlmodel.Field(default=None, nullable=True)
    analysis_status: Optional[Status] = sqlmodel.Field(default=None, nullable=True)

    video_length: Optional[int] = sqlmodel.Field(default=None, nullable=True)
    fps: Optional[int] = sqlmodel.Field(default=None, nullable=True)
    frames: Optional[int] = sqlmodel.Field(default=None, nullable=True)

    emotion: Optional[str] = sqlmodel.Field(default=None, nullable=True)
    intent: Optional[str] = sqlmodel.Field(default=None, nullable=True)
