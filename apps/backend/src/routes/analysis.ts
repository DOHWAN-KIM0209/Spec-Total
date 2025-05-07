import express from 'express';
import { PrismaClient } from '@prisma/client';

const router = express.Router();
const prisma = new PrismaClient();

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
router.get('/', async (req, res) => {
  try {
    const analyses = await prisma.analysis.findMany({
      include: {
        user: { select: { id: true, name: true } },
      },
      orderBy: { analysisReqTime: 'desc' },
    });

    res.json(analyses);
  } catch (err) {
    res.status(500).json({ error: '분석 조회 실패' });
  }
});

// 분석 기록 등록
router.post('/', async (req, res) => {
  try {
    const {
      userId,
      type,
      question,
      videoPath,
      thumbnailPath,
      keyword,
      setStartTime,
      analysisStartTime,
      analysisEndTime,
    } = req.body;

    const newAnalysis = await prisma.analysis.create({
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
  } catch (err) {
    res.status(500).json({ error: '분석 기록 등록 실패' });
  }
});

export default router;
