#!/bin/bash

echo "🔧 의존성 설치 및 실행 시작..."

# =============================
# Backend 설치 및 준비
# =============================
echo "📦 [1/3] Backend 의존성 설치 중..."
cd ./apps/backend || exit
npm install
cd - > /dev/null

# =============================
# AI 설치 (가상환경 + poetry)
# =============================
echo "📦 [2/3] AI 가상환경 및 Poetry 설치 중..."
cd ./apps/ai || exit

# 가상환경이 없으면 생성
if [ ! -d ".venv" ]; then
  echo "📁 가상환경 생성 중..."
  python3 -m venv .venv
fi

# 가상환경 활성화
source .venv/bin/activate

# poetry 설치 (존재 안하면 설치)
if ! command -v poetry &> /dev/null; then
  echo "📥 poetry 설치 중..."
  pip install poetry
fi

# 의존성 설치
poetry config virtualenvs.create false
poetry install --without dev,ai

deactivate
cd - > /dev/null

# =============================
# Frontend 설치 및 빌드
# =============================
echo "📦 [3/3] Web 의존성 설치 및 빌드 중..."
cd ./apps/web || exit
npm install --legacy-peer-deps
npm run build
cd - > /dev/null

# =============================
# Docker Compose 실행
# =============================
echo "🐳 Docker Compose 빌드 및 실행..."
docker compose up --build