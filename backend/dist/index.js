"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = __importDefault(require("express"));
const cors_1 = __importDefault(require("cors"));
const dotenv_1 = __importDefault(require("dotenv"));
const auth_1 = __importDefault(require("./src/routes/auth"));
const question_1 = __importDefault(require("./src/routes/question"));
const script_1 = __importDefault(require("./src/routes/script"));
const resume_1 = __importDefault(require("./src/routes/resume"));
const commonQuestion_1 = __importDefault(require("./src/routes/commonQuestion"));
const commonScript_1 = __importDefault(require("./src/routes/commonScript"));
const commonKeyword_1 = __importDefault(require("./src/routes/commonKeyword"));
const resumeQuestion_1 = __importDefault(require("./src/routes/resumeQuestion"));
const resumeScript_1 = __importDefault(require("./src/routes/resumeScript"));
const resumeKeyword_1 = __importDefault(require("./src/routes/resumeKeyword"));
const analysis_1 = __importDefault(require("./src/routes/analysis"));
const swagger_1 = require("./swagger");
const upload_1 = __importDefault(require("./src/routes/upload"));
const path_1 = __importDefault(require("path"));
const uploadAnalysis_1 = __importDefault(require("./src/routes/uploadAnalysis"));
dotenv_1.default.config();
const app = (0, express_1.default)();
const PORT = Number(process.env.PORT) || 4000;
const BASE_URL = process.env.BASE_URL || `https://${process.env.RAILWAY_STATIC_URL || 'localhost:4000'}`;
app.use((0, cors_1.default)());
app.use(express_1.default.json());
app.use('/auth', auth_1.default);
app.use('/questions', question_1.default);
app.use('/scripts', script_1.default);
app.use('/resumes', resume_1.default);
app.use('/common-questions', commonQuestion_1.default);
app.use('/common-scripts', commonScript_1.default);
app.use('/common-keywords', commonKeyword_1.default);
app.use('/resume-questions', resumeQuestion_1.default);
app.use('/resume-scripts', resumeScript_1.default);
app.use('/resume-keywords', resumeKeyword_1.default);
app.use('/analyses', analysis_1.default);
app.use('/upload', upload_1.default);
app.use('/uploads', express_1.default.static(path_1.default.join(__dirname, 'uploads')));
app.use('/public', express_1.default.static(path_1.default.join(__dirname, 'public')));
app.use('/upload-analysis', uploadAnalysis_1.default);
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
(0, swagger_1.setupSwagger)(app);
app.listen(PORT, '0.0.0.0', () => {
    console.log(`✅ Server running on ${BASE_URL}`);
});
