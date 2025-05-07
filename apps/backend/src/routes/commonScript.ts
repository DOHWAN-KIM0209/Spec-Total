import express from 'express';
import { PrismaClient } from '@prisma/client';

const router = express.Router();
const prisma = new PrismaClient();

// 공통 스크립트 전체 조회
/**
 * @swagger
 * tags:
 *   name: CommonScript
 *   description: 공통 스크립트 API
 */

/**
 * @swagger
 * /common-scripts:
 *   get:
 *     summary: 공통 스크립트 전체 조회
 *     tags: [CommonScript]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: 스크립트 목록 반환
 */

/**
 * @swagger
 * /common-scripts:
 *   post:
 *     summary: 공통 스크립트 등록
 *     tags: [CommonScript]
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
 *               script:
 *                 type: string
 *     responses:
 *       200:
 *         description: 등록된 스크립트 반환
 */
router.get('/', async (req, res) => {
  try {
    const scripts = await prisma.commonScript.findMany({
      include: {
        user: { select: { id: true, name: true } },
        question: { select: { id: true, question: true } },
      },
      orderBy: { createdTime: 'desc' },
    });
    res.json(scripts);
  } catch (err) {
    res.status(500).json({ error: '스크립트 조회 실패' });
  }
});

// 공통 스크립트 등록
router.post('/', async (req, res) => {
  try {
    const { userId, questionId, script } = req.body;

    const created = await prisma.commonScript.create({
      data: {
        userId: BigInt(userId),
        questionId: BigInt(questionId),
        script,
      },
    });

    res.json(created);
  } catch (err) {
    res.status(500).json({ error: '스크립트 등록 실패' });
  }
});

export default router;
