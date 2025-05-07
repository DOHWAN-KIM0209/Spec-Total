import { Request, Response, RequestHandler } from 'express';
import { PrismaClient } from '@prisma/client';
import bcrypt from 'bcrypt';
import jwt from 'jsonwebtoken';
import axios from 'axios';

const prisma = new PrismaClient();
const JWT_SECRET = process.env.JWT_SECRET || 'supersecret';
const KAKAO_CLIENT_ID = process.env.KAKAO_CLIENT_ID!;
const KAKAO_REDIRECT_URI = process.env.KAKAO_REDIRECT_URI!;

export const signup = async (req: Request, res: Response): Promise<void> => {
  const { email, password, name, role } = req.body;

  try {
    const hashedPassword = await bcrypt.hash(password, 10);

    const user = await prisma.user.create({
      data: {
        email,
        password: hashedPassword,
        name,
        role,
      },
    });

    res.status(201).json({ message: '회원가입 완료!', user });
  } catch (error) {
    res.status(400).json({ error: '이미 가입된 이메일일 수 있어요!' });
  }
};

export const login = async (req: Request, res: Response): Promise<void> => {
  const { email, password } = req.body;

  const user = await prisma.user.findUnique({ where: { email } });
  if (!user) {
    res.status(404).json({ error: '가입된 이메일이 아닙니다.' });
    return;
  }

  const isValid = await bcrypt.compare(password, user.password);
  if (!isValid) {
    res.status(401).json({ error: '비밀번호가 틀렸습니다.' });
    return;
  }

  const token = jwt.sign({ userId: user.id.toString() }, JWT_SECRET, { expiresIn: '7d' });
  res.json({ message: '로그인 성공', token });
};

export const me = async (req: Request, res: Response): Promise<void> => {
  const userId = BigInt((req as any).userId);

  const user = await prisma.user.findUnique({ where: { id: userId } });
  if (!user) {
    res.status(404).json({ error: '유저 정보를 찾을 수 없습니다.' });
    return;
  }

  res.json({
    user: {
      ...user,
      id: user.id.toString(),
    },
  });
};

export const kakaoLogin: RequestHandler = (req: Request, res: Response) => {
  const kakaoAuthURL = `https://kauth.kakao.com/oauth/authorize?response_type=code&client_id=${KAKAO_CLIENT_ID}&redirect_uri=${KAKAO_REDIRECT_URI}`;
  res.redirect(kakaoAuthURL);
};

export const kakaoCallback: RequestHandler = async (req: Request, res: Response) => {
  const code = req.query.code as string;

  const tokenRes = await axios.post(
    'https://kauth.kakao.com/oauth/token',
    null,
    {
      params: {
        grant_type: 'authorization_code',
        client_id: KAKAO_CLIENT_ID,
        redirect_uri: KAKAO_REDIRECT_URI,
        code,
      },
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    }
  );

  const access_token = tokenRes.data.access_token;

  const profileRes = await axios.get('https://kapi.kakao.com/v2/user/me', {
    headers: {
      Authorization: `Bearer ${access_token}`,
    },
  });

  const kakaoId = profileRes.data.id;
  const kakaoEmail = profileRes.data.kakao_account?.email;

  let social = await prisma.social.findFirst({
    where: {
      provider: 'kakao',
      providerId: kakaoId,
    },
    include: { user: true },
  });

  let user;

  if (social) {
    user = social.user;
  } else {
    user = await prisma.user.create({
      data: {
        email: kakaoEmail || `${kakaoId}@kakao.com`,
        name: '카카오유저',
        password: '',
        role: 'user',
      },
    });

    await prisma.social.create({
      data: {
        userId: user.id,
        provider: 'kakao',
        providerId: kakaoId,
      },
    });
  }

  const token = jwt.sign({ userId: user.id.toString() }, JWT_SECRET, { expiresIn: '7d' });
  res.json({ message: '카카오 로그인 완료', token });
};

export const logout: RequestHandler = async (req: Request, res: Response) => {
  res.json({ message: '로그아웃 처리: 클라이언트에서 토큰 삭제하세요' });
};
