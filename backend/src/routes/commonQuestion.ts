import express from 'express';
import { PrismaClient } from '@prisma/client';

const router = express.Router();
const prisma = new PrismaClient();

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
router.get('/', async (req, res) => {
  try {
    const questions = await prisma.commonQuestion.findMany({
      include: { category: true },
      orderBy: { createdTime: 'desc' },
    });
    res.json(questions);
  } catch (err) {
    res.status(500).json({ error: '질문 조회 실패' });
  }
});

// 공통 질문 추가
router.post('/', async (req, res) => {
  try {
    const { categoryId, question } = req.body;

    const newQuestion = await prisma.commonQuestion.create({
      data: {
        categoryId: BigInt(categoryId),
        question,
      },
    });

    res.json(newQuestion);
  } catch (err) {
    res.status(500).json({ error: '질문 생성 실패' });
  }
});

export default router;
