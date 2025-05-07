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
const client_1 = require("@prisma/client");
const router = express_1.default.Router();
const prisma = new client_1.PrismaClient();
// 분석 기록 전체 조회
/**
 * @swagger
 * tags:
 *   name: Analysis
 *   description: 면접 분석 기록 API
 */
/**
 * @swagger
 * /analyses:
 *   get:
 *     summary: 분석 기록 전체 조회 (인증 필요)
 *     tags: [Analysis]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: 분석 목록 반환
 */
/**
 * @swagger
 * /analyses:
 *   post:
 *     summary: 분석 기록 등록
 *     tags: [Analysis]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
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
 *               videoPath:
 *                 type: string
 *               thumbnailPath:
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
 *     responses:
 *       200:
 *         description: 등록된 분석 정보 반환
 */
router.get('/', (req, res) => __awaiter(void 0, void 0, void 0, function* () {
    try {
        const analyses = yield prisma.analysis.findMany({
            include: {
                user: { select: { id: true, name: true } },
            },
            orderBy: { analysisReqTime: 'desc' },
        });
        res.json(analyses);
    }
    catch (err) {
        res.status(500).json({ error: '분석 조회 실패' });
    }
}));
// 분석 기록 등록
router.post('/', (req, res) => __awaiter(void 0, void 0, void 0, function* () {
    try {
        const { userId, type, question, videoPath, thumbnailPath, keyword, setStartTime, analysisStartTime, analysisEndTime, } = req.body;
        const newAnalysis = yield prisma.analysis.create({
            data: {
                userId: BigInt(userId),
                type,
                question,
                videoPath,
                thumbnailPath,
                keyword,
                setStartTime: new Date(setStartTime),
                analysisStartTime: new Date(analysisStartTime),
                analysisEndTime: analysisEndTime ? new Date(analysisEndTime) : null,
            },
        });
        res.json(newAnalysis);
    }
    catch (err) {
        res.status(500).json({ error: '분석 기록 등록 실패' });
    }
}));
exports.default = router;
