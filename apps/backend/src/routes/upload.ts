import express from 'express';
import multer from 'multer';
import path from 'path';
import fs from 'fs';

const router = express.Router();

// 업로드 경로 설정
const uploadDir = path.join(__dirname, '../uploads');
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir);
}

// multer 설정
const storage = multer.diskStorage({
  destination: (_, __, cb) => cb(null, uploadDir),
  filename: (_, file, cb) => {
    const ext = path.extname(file.originalname);
    const base = path.basename(file.originalname, ext);
    const filename = `${base}-${Date.now()}${ext}`;
    cb(null, filename);
  },
});

const upload = multer({ storage });

/**
 * @swagger
 * /upload:
 *   post:
 *     summary: 영상/썸네일 파일 업로드
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
 *               video:
 *                 type: string
 *                 format: binary
 *               thumbnail:
 *                 type: string
 *                 format: binary
 *     responses:
 *       200:
 *         description: 업로드 성공
 */
router.post('/', upload.fields([
  { name: 'video', maxCount: 1 },
  { name: 'thumbnail', maxCount: 1 },
]), (req, res) => {
  const files = req.files as {
    [fieldname: string]: Express.Multer.File[];
  };
  const video = files['video']?.[0];
  const thumbnail = files['thumbnail']?.[0];

  res.json({
    videoPath: video ? `/uploads/${video.filename}` : null,
    thumbnailPath: thumbnail ? `/uploads/${thumbnail.filename}` : null,
  });
});

export default router;
