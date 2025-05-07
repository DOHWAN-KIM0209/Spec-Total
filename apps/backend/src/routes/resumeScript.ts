import express from 'express';
import { PrismaClient } from '@prisma/client';

const router = express.Router();
const prisma = new PrismaClient();

// 이력서 기반 스크립트 전체 조회
/**
 * @swagger
 * tags:
 *   name: ResumeScript
 *   description: 이력서 기반 스크립트 API
 */

/**
 * @swagger
 * /resume-scripts:
 *   get:
 *     summary: 이력서 스크립트 전체 조회
 *     tags: [ResumeScript]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: 스크립트 목록 반환
 */

/**
 * @swagger
 * /resume-scripts:
 *   post:
 *     summary: 이력서 스크립트 등록
 *     tags: [ResumeScript]
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
 *               script:
 *                 type: string
 *     responses:
 *       200:
 *         description: 등록된 스크립트 반환
 */
router.get('/', async (req, res) => {
  try {
    const scripts = await prisma.resumeScript.findMany({
      include: {
        user: { select: { id: true, name: true } },
        resume: { select: { id: true, name: true } },
      },
      orderBy: { createdTime: 'desc' },
    });

    res.json(scripts);
  } catch (err) {
    res.status(500).json({ error: '이력서 스크립트 조회 실패' });
  }
});

// 이력서 기반 스크립트 등록
router.post('/', async (req, res) => {
  try {
    const { userId, questionId, resumeId, script } = req.body;

    const newScript = await prisma.resumeScript.create({
      data: {
        userId: BigInt(userId),
        questionId: BigInt(questionId),
        resumeId: BigInt(resumeId),
        script,
      },
    });

    res.json(newScript);
  } catch (err) {
    res.status(500).json({ error: '이력서 스크립트 등록 실패' });
  }
});

export default router;
