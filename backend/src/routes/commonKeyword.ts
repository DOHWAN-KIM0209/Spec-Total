import express from 'express';
import { PrismaClient } from '@prisma/client';

const router = express.Router();
const prisma = new PrismaClient();

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
router.get('/', async (req, res) => {
  try {
    const keywords = await prisma.commonKeyword.findMany({
      include: {
        user: { select: { id: true, name: true } },
        question: { select: { id: true, question: true } },
      },
      orderBy: { createdTime: 'desc' },
    });
    res.json(keywords);
  } catch (err) {
    res.status(500).json({ error: '키워드 조회 실패' });
  }
});

// 공통 키워드 등록
router.post('/', async (req, res) => {
  try {
    const { userId, questionId, keyword } = req.body;

    const created = await prisma.commonKeyword.create({
      data: {
        userId: BigInt(userId),
        questionId: BigInt(questionId),
        keyword,
      },
    });

    res.json(created);
  } catch (err) {
    res.status(500).json({ error: '키워드 등록 실패' });
  }
});

export default router;
