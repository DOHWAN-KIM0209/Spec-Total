import uvicorn
from common.exception import *
import openai
import os
from dotenv import load_dotenv
from fastapi import FastAPI, status, HTTPException, UploadFile, File
from loguru import logger
from routes import analysis
from io import BytesIO
from core.logger import load_config
from core.settings import settings
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware  # CORS 미들웨어 import

# PDF 텍스트 추출용
import pdfplumber

# .env 파일 로드
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
print("🔑 OpenAI KEY:", openai.api_key)

app = FastAPI(
    title="[spec] API-AI docs",
    description=""" 
    ## **spec**
    Give some guidance for the job interview based on AI
    - This is a project of D102, SSAFY 10th.
    - Project description on [Notion](https://lshhh.notion.site/preview-AI-3590d9e05aa447c6b92d8f65639632ff?pvs=74)

    Author:
    - Kim kyungran <cookie3478@naver.com>, Leader of D102
    """,
    root_path="/ai",
    openapi_tags=[{"name": "1. analysis", "description": "면접 영상 분석 관련 API"},
                  {"name": "2. interview", "description": "면접 질문 생성 관련 API"}],
)

# CORS 설정 (React와 연결을 위해 필요)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React 앱 주소
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메소드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

# Add routers(controllers)
app.include_router(
    analysis.router,
    tags=["1. analysis"],
    prefix="/analysis",
)

# Add custom exception handlers
exception_handlers = {
    RequestValidationError: custom_validation_error_handler,
    FileNotFoundError: custom_file_not_fount_error_handler,
}
for exc, handler in exception_handlers.items():
    app.add_exception_handler(exc, handler)

class ResumeQuestionRequest(BaseModel):
    resume: str

class FollowUpQuestionRequest(BaseModel):
    previous_question: str
    user_answer: str

# ✅ 1. 공통 면접 질문 생성 API
tags = ["2. interview"]
@app.get("/interview/common-questions", tags=["공통 면접 질문"])
async def generate_common_questions():
    """
    기본적인 공통 면접 질문 리스트를 생성하는 API.
    """
    common_questions = [
        "자기소개 부탁드립니다.",
        "본인의 강점과 약점은 무엇인가요?",
        "지원 동기는 무엇인가요?",
        "5년 후 본인의 모습을 어떻게 생각하나요?",
        "팀 프로젝트 경험에 대해 이야기해주세요.",
        "문제를 해결했던 경험을 알려주세요.",
        "갈등 상황에서 어떻게 대처했나요?",
        "최근 관심 있는 IT 트렌드는 무엇인가요?",
        "압박면접 상황이라면 어떻게 대응하시겠습니까?",
        "왜 우리 회사를 선택했나요?"
    ]
    return {"common_questions": common_questions}

# ✅ pdfplumber 기반 이력서 질문 생성 API
@app.post("/interview/resume-question", tags=tags)
async def generate_resume_question(file: UploadFile = File(...)):
    """
    이력서를 PDF 형식으로 업로드하면 OpenAI를 통해 면접 질문을 생성하는 API.
    pdfplumber를 통해 높은 정확도로 텍스트 추출.
    """
    try:
        pdf_bytes = await file.read()

        # pdfplumber로 텍스트 추출
        text = ""
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        if not text.strip():
            raise HTTPException(status_code=400, detail="PDF에서 텍스트를 추출할 수 없습니다.")

        print("📥 추출된 이력서 텍스트:", text)

        # GPT에 질문 요청
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "system",
                "content": "이력서를 분석해서, 이력서에 작성된 언어가 한국어면 한국어로만, 영어면 영어로만 된 자연스럽고 적절한 면접 질문을 10가지 리스트로 생성하세요."
            }, {
                "role": "user",
                "content": f"이력서:\n{text}"
            }]
        )

        return {"question": response["choices"][0]["message"]["content"]}

    except Exception as e:
        print("🔥 텍스트 추출 또는 OpenAI 에러:", e)
        raise HTTPException(status_code=500, detail=str(e))

# ✅ 꼬리질문 생성 API
@app.post("/interview/follow-up", tags=tags)
async def generate_follow_up_question(request: FollowUpQuestionRequest):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "system",
                "content": "질문과 사용자의 대답을 참고하여 꼬리질문을 생성하되, 사용자의 대답이 한국어면 한국어로, 영어면 영어로 자연스럽고 적절한 꼬리질문을 생성하세요."
            }, {
                "role": "user",
                "content": f"질문: {request.previous_question}\n대답: {request.user_answer}"
            }]
        )
        return {"follow_up_question": response["choices"][0]["message"]["content"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 서버 Health Check
@app.get(
    "/",
    summary="서버 Health Check",
    status_code=status.HTTP_200_OK,
)
def health_check():
    """서버가 실행 중이면 'ok' 반환"""
    return "ok"

if __name__ == "__main__":
    RUN_SERVER_MSG = "Run server"
    if settings.DEBUG:
        RUN_SERVER_MSG = f"NOTE!!! {RUN_SERVER_MSG} as DEBUG mode"
    logger.info(RUN_SERVER_MSG)

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_config=load_config(),
    )
