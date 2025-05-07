"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = __importDefault(require("express"));
const multer_1 = __importDefault(require("multer"));
const path_1 = __importDefault(require("path"));
const fs_1 = __importDefault(require("fs"));
const router = express_1.default.Router();
// 업로드 경로 설정
const uploadDir = path_1.default.join(__dirname, '../uploads');
if (!fs_1.default.existsSync(uploadDir)) {
    fs_1.default.mkdirSync(uploadDir);
}
// multer 설정
const storage = multer_1.default.diskStorage({
    destination: (_, __, cb) => cb(null, uploadDir),
    filename: (_, file, cb) => {
        const ext = path_1.default.extname(file.originalname);
        const base = path_1.default.basename(file.originalname, ext);
        const filename = `${base}-${Date.now()}${ext}`;
        cb(null, filename);
    },
});
const upload = (0, multer_1.default)({ storage });
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
    var _a, _b;
    const files = req.files;
    const video = (_a = files['video']) === null || _a === void 0 ? void 0 : _a[0];
    const thumbnail = (_b = files['thumbnail']) === null || _b === void 0 ? void 0 : _b[0];
    res.json({
        videoPath: video ? `/uploads/${video.filename}` : null,
        thumbnailPath: thumbnail ? `/uploads/${thumbnail.filename}` : null,
    });
});
exports.default = router;
