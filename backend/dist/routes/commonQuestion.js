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
// 공통 질문 목록 조회
/**
 * @swagger
 * tags:
 *   name: CommonQuestion
 *   description: 공통 질문 API
 */
/**
 * @swagger
 * /common-questions:
 *   get:
 *     summary: 공통 질문 전체 조회
 *     tags: [CommonQuestion]
 *     responses:
 *       200:
 *         description: 질문 목록 반환
 */
/**
 * @swagger
 * /common-questions:
 *   post:
 *     summary: 공통 질문 등록
 *     tags: [CommonQuestion]
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               categoryId:
 *                 type: integer
 *               question:
 *                 type: string
 *     responses:
 *       200:
 *         description: 등록된 질문 반환
 */
router.get('/', (req, res) => __awaiter(void 0, void 0, void 0, function* () {
    try {
        const questions = yield prisma.commonQuestion.findMany({
            include: { category: true },
            orderBy: { createdTime: 'desc' },
        });
        res.json(questions);
    }
    catch (err) {
        res.status(500).json({ error: '질문 조회 실패' });
    }
}));
// 공통 질문 추가
router.post('/', (req, res) => __awaiter(void 0, void 0, void 0, function* () {
    try {
        const { categoryId, question } = req.body;
        const newQuestion = yield prisma.commonQuestion.create({
            data: {
                categoryId: BigInt(categoryId),
                question,
            },
        });
        res.json(newQuestion);
    }
    catch (err) {
        res.status(500).json({ error: '질문 생성 실패' });
    }
}));
exports.default = router;
