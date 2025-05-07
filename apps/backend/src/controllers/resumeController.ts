import { Request, Response } from 'express';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export const uploadResume = async (req: Request, res: Response) => {
  const { userId, name, filePath } = req.body;

  try {
    const resume = await prisma.resume.create({
      data: {
        userId,
        name,
        filePath
      }
    });
    res.status(201).json({ message: '이력서 등록 완료', data: resume });
  } catch (error) {
    res.status(400).json({ error: '이력서 등록 실패', details: error });
  }
};

export const getUserResumes = async (req: Request, res: Response) => {
  const userId = Number(req.params.userId);

  const resumes = await prisma.resume.findMany({
    where: { userId }
  });

  res.json(resumes);
};
