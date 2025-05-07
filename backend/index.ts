import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import authRoutes from './src/routes/auth';
import questionRoutes from './src/routes/question';
import scriptRoutes from './src/routes/script';
import resumeRoutes from './src/routes/resume';
import commonQuestionRouter from './src/routes/commonQuestion';
import commonScriptRouter from './src/routes/commonScript';
import commonKeywordRouter from './src/routes/commonKeyword';
import resumeQuestionRouter from './src/routes/resumeQuestion';
import resumeScriptRouter from './src/routes/resumeScript';
import resumeKeywordRouter from './src/routes/resumeKeyword';
import analysisRouter from './src/routes/analysis';
import { setupSwagger } from './swagger';
import uploadRouter from './src/routes/upload';
import path from 'path';
import uploadAnalysisRouter from './src/routes/uploadAnalysis';

dotenv.config();

const app = express();
const PORT = Number(process.env.PORT) || 4000;
const BASE_URL = process.env.BASE_URL || `https://${process.env.RAILWAY_STATIC_URL || 'localhost:4000'}`;

app.use(cors());
app.use(express.json());

app.use('/auth', authRoutes);
app.use('/questions', questionRoutes);
app.use('/scripts', scriptRoutes);
app.use('/resumes', resumeRoutes);
app.use('/common-questions', commonQuestionRouter);
app.use('/common-scripts', commonScriptRouter);
app.use('/common-keywords', commonKeywordRouter);
app.use('/resume-questions', resumeQuestionRouter);
app.use('/resume-scripts', resumeScriptRouter);
app.use('/resume-keywords', resumeKeywordRouter);
app.use('/analyses', analysisRouter);
app.use('/upload', uploadRouter);
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));
app.use('/public', express.static(path.join(__dirname, 'public')));
app.use('/upload-analysis', uploadAnalysisRouter);

app.get('/', (req, res) => {
  console.log('✅ 루트 경로 요청 들어옴');
  res.send(`
    <!DOCTYPE html>
    <html lang="ko">
      <head>
        <meta charset="UTF-8" />
        <meta property="og:title" content="AI 면접 분석 서비스" />
        <meta property="og:description" content="AI가 당신의 면접을 분석해드립니다." />
        <meta property="og:image" content="${BASE_URL}/og-image.jpg" />
        <meta property="og:url" content="${BASE_URL}" />
        <title>AI 면접 백엔드 연결 성공</title>
      </head>
      <body>
        <h1>✅ 백엔드 서버 연결 성공!</h1>
        <img src="/og-image.jpg" width="300" />
      </body>
    </html>
  `);
});

setupSwagger(app);

app.listen(PORT, '0.0.0.0', () => {
  console.log(`✅ Server running on ${BASE_URL}`);
});
