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
// 공통 키워드 목록 조회
/**
 * @swagger
 * /common-keywords:
 *   get:
 *     summary: 공통 키워드 전체 조회
 *     tags: [CommonKeyword]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: 키워드 목록 반환
 */
/**
 * @swagger
 * /common-keywords:
 *   post:
 *     summary: 공통 키워드 등록
 *     tags: [CommonKeyword]
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
 *               questionId:
 *                 type: integer
 *               keyword:
 *                 type: string
 *     responses:
 *       200:
 *         description: 등록된 키워드 반환
 */
router.get('/', (req, res) => __awaiter(void 0, void 0, void 0, function* () {
    try {
        const keywords = yield prisma.commonKeyword.findMany({
            include: {
                user: { select: { id: true, name: true } },
                question: { select: { id: true, question: true } },
            },
            orderBy: { createdTime: 'desc' },
        });
        res.json(keywords);
    }
    catch (err) {
        res.status(500).json({ error: '키워드 조회 실패' });
    }
}));
// 공통 키워드 등록
router.post('/', (req, res) => __awaiter(void 0, void 0, void 0, function* () {
    try {
        const { userId, questionId, keyword } = req.body;
        const created = yield prisma.commonKeyword.create({
            data: {
                userId: BigInt(userId),
                questionId: BigInt(questionId),
                keyword,
            },
        });
        res.json(created);
    }
    catch (err) {
        res.status(500).json({ error: '키워드 등록 실패' });
    }
}));
exports.default = router;
