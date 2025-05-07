import express from 'express';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import { PrismaClient } from '@prisma/client';

const router = express.Router();
const prisma = new PrismaClient();

// 업로드 디렉토리 설정
const uploadDir = path.join(__dirname, '../uploads');
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir);
}

// Multer 설정
const storage = multer.diskStorage({
  destination: (_, __, cb) => cb(null, uploadDir),
  filename: (_, file, cb) => {
    const ext = path.extname(file.originalname);
    const base = path.basename(file.originalname, ext);
    cb(null, `${base}-${Date.now()}${ext}`);
  },
});
const upload = multer({ storage });

/**
 * @swagger
 * /upload-analysis:
 *   post:
 *     summary: 영상 업로드 및 분석 기록 등록
 *     tags: [Upload]
 *     consumes:
 *       - multipart/form-data
 *     requestBody:
 *       required: true
 *       content:
 *         multipart/form-data:
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
 *               video:
 *                 type: string
 *                 format: binary
 *               thumbnail:
 *                 type: string
 *                 format: binary
 *     responses:
 *       200:
 *         description: 분석 기록 저장됨
 */
router.post(
  '/',
  upload.fields([
    { name: 'video', maxCount: 1 },
    { name: 'thumbnail', maxCount: 1 },
  ]),
  async (req, res) => {
    try {
      const {
        userId, type, question, keyword,
        setStartTime, analysisStartTime, analysisEndTime
      } = req.body;

      const files = req.files as {
        [fieldname: string]: Express.Multer.File[];
      };
      const video = files['video']?.[0];
      const thumbnail = files['thumbnail']?.[0];

      const newAnalysis = await prisma.analysis.create({
        data: {
          userId: BigInt(userId),
          type,
          question,
          keyword,
          videoPath: video ? `/uploads/${video.filename}` : '',
          thumbnailPath: thumbnail ? `/uploads/${thumbnail.filename}` : '',
          setStartTime: new Date(setStartTime),
          analysisReqTime: new Date(),
          analysisStartTime: new Date(analysisStartTime),
          analysisEndTime: analysisEndTime ? new Date(analysisEndTime) : null,
        },
      });

      res.json(newAnalysis);
    } catch (error) {
      console.error(error);
      res.status(500).json({ error: '분석 업로드 실패' });
    }
  }
);

export default router;
