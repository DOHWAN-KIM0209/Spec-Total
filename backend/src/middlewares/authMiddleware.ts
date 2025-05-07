import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';

const JWT_SECRET = process.env.JWT_SECRET || 'supersecret';

export const authMiddleware = (req: Request, res: Response, next: NextFunction): void => {
  try {
    const authHeader = req.headers.authorization;
    if (!authHeader) {
      res.status(401).json({ error: '토큰이 없습니다.' });
      return;
    }

    const token = authHeader.split(' ')[1];
    const decoded = jwt.verify(token, JWT_SECRET) as { userId: string | number };

    (req as any).userId = decoded.userId;
    next();
  } catch (err: any) {
    res.status(401).json({
      error: '유효하지 않은 토큰입니다.',
      detail: err.message,
    });
  }
};
