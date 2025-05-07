# 🎯 Interview-AI Backend 연동 안내

이 프로젝트는 Railway에 배포된 백엔드 서버를 사용하여 AI 면접 분석 기능을 제공합니다. 아래 가이드를 통해 프론트엔드에서 어떻게 연동하면 되는지 설명합니다.

---

## 📍 1. API 서버 주소

> Base URL:
```
https://interview-ai-backend.up.railway.app
```

---

## 🚀 2. API 엔드포인트 예시

| 목적 | 메서드 | 경로 |
|------|--------|------|
| 로그인 | POST | `/auth/login` |
| 회원가입 | POST | `/auth/register` |
| 면접 질문 불러오기 | GET | `/questions` |
| 스크립트 제출 | POST | `/scripts` |
| 분석 요청 업로드 | POST | `/upload-analysis` |
| 정적 이미지 접근 (OG 이미지 등) | GET | `/uploads/og-image.png` |

---

## 🖼 3. 메타태그(Open Graph) 이미지 연동

```html
<meta property="og:title" content="AI 면접 분석 서비스" />
<meta property="og:description" content="AI가 당신의 면접을 분석해드립니다." />
<meta property="og:image" content="https://interview-ai-backend.up.railway.app/uploads/og-image.png" />
<meta property="og:url" content="https://interview-ai-backend.up.railway.app" />
```

---

## 📦 4. 프론트엔드 코드에서 연동 방법 예시 (axios)

```ts
import axios from 'axios';

const api = axios.create({
  baseURL: 'https://interview-ai-backend.up.railway.app',
});

export default api;
```

---

## 📚 5. API 명세 (Swagger)

> 문서 확인 URL:
```
https://interview-ai-backend.up.railway.app/api-docs
```

---

## ✅ 기타 참고사항

- 모든 요청은 CORS 허용되어 있음
- 이미지 등 정적 리소스는 `/uploads` 경로에서 접근 가능
- 백엔드가 Express 기반이며, Railway에서 자동 배포됨