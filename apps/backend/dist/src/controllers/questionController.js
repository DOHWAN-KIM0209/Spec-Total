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
Object.defineProperty(exports, "__esModule", { value: true });
exports.getQuestionsByCategory = exports.getAllQuestions = exports.createQuestion = void 0;
const client_1 = require("@prisma/client");
const prisma = new client_1.PrismaClient();
// 공통 질문 등록
const createQuestion = (req, res) => __awaiter(void 0, void 0, void 0, function* () {
    const { category, question } = req.body;
    let foundCategory = yield prisma.commonCategory.findUnique({ where: { category } });
    if (!foundCategory) {
        foundCategory = yield prisma.commonCategory.create({ data: { category } });
    }
    const created = yield prisma.commonQuestion.create({
        data: {
            categoryId: foundCategory.id,
            question,
        },
    });
    res.status(201).json(created);
});
exports.createQuestion = createQuestion;
// 전체 질문 조회
const getAllQuestions = (req, res) => __awaiter(void 0, void 0, void 0, function* () {
    const questions = yield prisma.commonQuestion.findMany({
        include: { category: true },
    });
    res.json(questions);
});
exports.getAllQuestions = getAllQuestions;
// 카테고리별 질문 조회
const getQuestionsByCategory = (req, res) => __awaiter(void 0, void 0, void 0, function* () {
    const { category } = req.params;
    const foundCategory = yield prisma.commonCategory.findUnique({ where: { category } });
    if (!foundCategory) {
        res.status(404).json({ error: '카테고리 없음' });
        return;
    }
    const questions = yield prisma.commonQuestion.findMany({
        where: { categoryId: foundCategory.id },
    });
    res.json(questions);
});
exports.getQuestionsByCategory = getQuestionsByCategory;
