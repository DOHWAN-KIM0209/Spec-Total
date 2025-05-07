"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = __importDefault(require("express"));
const multer_1 = __importDefault(require("multer"));
const path_1 = __importDefault(require("path"));
const fs_1 = __importDefault(require("fs"));
const client_1 = require("@prisma/client");
const router = express_1.default.Router();
const prisma = new client_1.PrismaClient();
// 업로드 디렉토리 설정
const uploadDir = path_1.default.join(__dirname, '../uploads');
if (!fs_1.default.existsSync(uploadDir)) {
    fs_1.default.mkdirSync(uploadDir);
}
// Multer 설정
const storage = multer_1.default.diskStorage({
    destination: (_, __, cb) => cb(null, uploadDir),
    filename: (_, file, cb) => {
        const ext = path_1.default.extname(file.originalname);
        const base = path_1.default.basename(file.originalname, ext);
        cb(null, `${base}-${Date.now()}${ext}`);
    },
});
const upload = (0, multer_1.default)({ storage });
/**
 * @swagger
 * /upload-analysis:
 *   post:
 *     summary: 영상 업로드 및 분석 기록 등록
 *     tags: [Upload]
 *     consumes:
 *       - multipart/form-data
 *     requestBody:
 *       required: true
 *       content:
 *         multipart/form-data:
 *           schema:
 *             type: object
 *             properties:
 *               userId:
 *                 type: integer
 *               type:
 *                 type: string
 *                 example: "mock"
 *               question:
 *                 type: string
 *               keyword:
 *                 type: string
 *               setStartTime:
 *                 type: string
 *                 format: date-time
 *               analysisStartTime:
 *                 type: string
 *                 format: date-time
 *               analysisEndTime:
 *                 type: string
 *                 format: date-time
 *               video:
 *                 type: string
 *                 format: binary
 *               thumbnail:
 *                 type: string
 *                 format: binary
 *     responses:
 *       200:
 *         description: 분석 기록 저장됨
 */
router.post('/', upload.fields([
    { name: 'video', maxCount: 1 },
    { name: 'thumbnail', maxCount: 1 },
]), (req, res) => __awaiter(void 0, void 0, void 0, function* () {
    var _a, _b;
    try {
        const { userId, type, question, keyword, setStartTime, analysisStartTime, analysisEndTime } = req.body;
        const files = req.files;
        const video = (_a = files['video']) === null || _a === void 0 ? void 0 : _a[0];
        const thumbnail = (_b = files['thumbnail']) === null || _b === void 0 ? void 0 : _b[0];
        const newAnalysis = yield prisma.analysis.create({
            data: {
                userId: BigInt(userId),
                type,
                question,
                keyword,
                videoPath: video ? `/uploads/${video.filename}` : '',
                thumbnailPath: thumbnail ? `/uploads/${thumbnail.filename}` : '',
                setStartTime: new Date(setStartTime),
                analysisReqTime: new Date(),
                analysisStartTime: new Date(analysisStartTime),
                analysisEndTime: analysisEndTime ? new Date(analysisEndTime) : null,
            },
        });
        res.json(newAnalysis);
    }
    catch (error) {
        console.error(error);
        res.status(500).json({ error: '분석 업로드 실패' });
    }
}));
exports.default = router;
