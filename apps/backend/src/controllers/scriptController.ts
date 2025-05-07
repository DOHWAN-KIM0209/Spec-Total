import { Request, Response } from 'express';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export const createScript = async (req: Request, res: Response) => {
  const { userId, questionId, script } = req.body;

  try {
    const created = await prisma.commonScript.create({
      data: {
        userId,
        questionId,
        script,
      },
    });

    res.status(201).json({ message: '스크립트 저장 완료', data: created });
  } catch (err) {
    res.status(400).json({ error: '스크립트 저장 실패', details: err });
  }
};

export const getUserScripts = async (req: Request, res: Response) => {
  const userId = Number(req.params.userId);

  const scripts = await prisma.commonScript.findMany({
    where: { userId },
    include: {
      question: true,
    },
  });

  res.json(scripts);
};
