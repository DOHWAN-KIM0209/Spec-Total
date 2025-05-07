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
exports.getUserResumes = exports.uploadResume = void 0;
const client_1 = require("@prisma/client");
const prisma = new client_1.PrismaClient();
const uploadResume = (req, res) => __awaiter(void 0, void 0, void 0, function* () {
    const { userId, name, filePath } = req.body;
    try {
        const resume = yield prisma.resume.create({
            data: {
                userId,
                name,
                filePath
            }
        });
        res.status(201).json({ message: '이력서 등록 완료', data: resume });
    }
    catch (error) {
        res.status(400).json({ error: '이력서 등록 실패', details: error });
    }
});
exports.uploadResume = uploadResume;
const getUserResumes = (req, res) => __awaiter(void 0, void 0, void 0, function* () {
    const userId = Number(req.params.userId);
    const resumes = yield prisma.resume.findMany({
        where: { userId }
    });
    res.json(resumes);
});
exports.getUserResumes = getUserResumes;
