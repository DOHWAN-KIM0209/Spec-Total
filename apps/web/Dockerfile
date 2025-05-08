FROM nginx:latest

EXPOSE 3000

# 정적 빌드 파일을 nginx 루트로 복사
ADD ./dist /usr/share/nginx/html

# 기본 nginx.conf를 사용 (원하는 경우 커스텀 가능)
