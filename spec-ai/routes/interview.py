from fastapi import APIRouter
from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# API 키 확인 (테스트용)
print(f"🔑 Loaded OpenAI API Key: {openai.api_key}")

# FastAPI 라우터 생성
router = APIRouter()

# 요청 데이터 모델 정의
class ResumeRequest(BaseModel):
    resume_text: str  # 이력서 텍스트

@router.post("/questions", summary="이력서를 기반으로 면접 질문 생성")
def generate_question(req: ResumeRequest):
    """
    OpenAI API를 사용하여 이력서를 기반으로 면접 질문을 생성하는 API
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # 또는 gpt-3.5-turbo
            messages=[
                {"role": "system", "content": "You are an AI trained to generate interview questions."},
                {"role": "user", "content": f"Generate an interview question based on this resume: {req.resume_text}"}
            ]
        )
        question = response["choices"][0]["message"]["content"]
        return {"question": question}
    except Exception as e:
        return {"error": str(e)}
