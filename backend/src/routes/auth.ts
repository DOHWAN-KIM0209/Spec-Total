import express from 'express';
import { signup, login, me, logout } from '../controllers/authController';
import { authMiddleware } from '../middlewares/authMiddleware';
import { kakaoLogin, kakaoCallback } from '../controllers/authController';


const router = express.Router();

router.post('/signup', signup);
router.post('/login', login);
router.get('/me', authMiddleware, me);
router.get('/kakao', kakaoLogin);
router.get('/kakao/callback', kakaoCallback);
router.post('/logout', logout);

export default router;
