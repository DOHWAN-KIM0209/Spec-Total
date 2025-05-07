# database.py
from sqlmodel import create_engine, Session
from contextlib import contextmanager

DATABASE_URL = "sqlite:///./test.db"  # SQLite 데이터베이스 사용

# 데이터베이스 연결 엔진 생성
engine = create_engine(DATABASE_URL, echo=True)

# 세션 생성 함수
@contextmanager
def get_session():
    session = Session(engine)
    try:
        yield session
    finally:
        session.close()
