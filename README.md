# Spec-Total 모노레포 프로젝트

이 저장소는 Spec 프로젝트의 모노레포 구조로, 다음과 같은 세 가지 주요 애플리케이션을 포함합니다.

- **백엔드**: API 서버
- **AI**: AI 모델 및 추론 서비스
- **프론트엔드**: 사용자 인터페이스

## 📁 프로젝트 구조

Spec-Total/
├── apps/
│ ├── backend/ # 백엔드 API 서버
│ ├── ai/ # AI 모델 및 추론 서비스
│ └── web/ # 프론트엔드 웹 애플리케이션
├── README.md
├── docker-compose.yml
└── start.sh

## 🚀 시작하기

### 1. 의존성 설치

각 애플리케이션의 디렉토리로 이동하여 필요한 의존성을 설치합니다.

#### 백엔드

```bash
cd apps/backend
npm install
AI
bash
복사
편집
cd apps/ai
pip install -r requirements.txt
프론트엔드
bash
복사
편집
cd apps/web
npm install
2. 애플리케이션 실행
각 애플리케이션을 개별적으로 실행하거나, docker-compose를 사용하여 전체를 한 번에 실행할 수 있습니다.

개별 실행
백엔드: npm start

AI: python main.py (예시)

프론트엔드: npm run dev

전체 실행
bash
복사
편집
./start.sh
또는

bash
복사
편집
docker-compose up --build
🛠 기술 스택
백엔드: Node.js, Express

AI: Python, KoBERT

프론트엔드: React, Vite, Tailwind CSS

📄 라이선스
이 프로젝트는 MIT 라이선스를 따릅니다.