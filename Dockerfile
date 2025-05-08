# 루트 Dockerfile (Spec-Total/Dockerfile)
FROM node:18

WORKDIR /app
COPY ./apps/backend/package*.json ./
RUN npm install
COPY ./apps/backend .

EXPOSE 4000
CMD ["npm", "start"]
