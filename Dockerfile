FROM node:18

WORKDIR /app

# 1단계: package.json 먼저 복사
COPY ./apps/backend/package*.json ./

# 2단계: prisma schema도 복사
COPY ./apps/backend/prisma ./prisma

# 3단계: 의존성 설치
RUN npm install

# 4단계: 전체 백엔드 소스 복사
COPY ./apps/backend .

# 포트 오픈
EXPOSE 4000

# 실행
CMD ["npm", "start"]
