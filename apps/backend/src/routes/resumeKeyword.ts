import express from 'express';
import { PrismaClient } from '@prisma/client';

const router = express.Router();
const prisma = new PrismaClient();

// 이력서 키워드 전체 조회
/**
 * @swagger
 * /resume-keywords:
 *   get:
 *     summary: 이력서 키워드 전체 조회
 *     tags: [ResumeKeyword]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: 키워드 목록 반환
 */

/**
 * @swagger
 * /resume-keywords:
 *   post:
 *     summary: 이력서 키워드 등록
 *     tags: [ResumeKeyword]
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
 *               resumeId:
 *                 type: integer
 *               keyword:
 *                 type: string
 *     responses:
 *       200:
 *         description: 등록된 키워드 반환
 */
router.get('/', async (req, res) => {
  try {
    const keywords = await prisma.resumeKeyword.findMany({
      include: {
        user: { select: { id: true, name: true } },
        resume: { select: { id: true, name: true } },
      },
      orderBy: { createdTime: 'desc' },
    });

    res.json(keywords);
  } catch (err) {
    res.status(500).json({ error: '이력서 키워드 조회 실패' });
  }
});

// 이력서 키워드 등록
router.post('/', async (req, res) => {
  try {
    const { userId, questionId, resumeId, keyword } = req.body;

    const newKeyword = await prisma.resumeKeyword.create({
      data: {
        userId: BigInt(userId),
        questionId: BigInt(questionId),
        resumeId: BigInt(resumeId),
        keyword,
      },
    });

    res.json(newKeyword);
  } catch (err) {
    res.status(500).json({ error: '이력서 키워드 등록 실패' });
  }
});

export default router;
