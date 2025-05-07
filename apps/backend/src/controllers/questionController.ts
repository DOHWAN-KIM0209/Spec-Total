import { Request, Response } from 'express';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

// 공통 질문 등록
export const createQuestion = async (req: Request, res: Response): Promise<void> => {
  const { category, question } = req.body;

  let foundCategory = await prisma.commonCategory.findUnique({ where: { category } });

  if (!foundCategory) {
    foundCategory = await prisma.commonCategory.create({ data: { category } });
  }

  const created = await prisma.commonQuestion.create({
    data: {
      categoryId: foundCategory.id,
      question,
    },
  });

  res.status(201).json(created);
};

// 전체 질문 조회
export const getAllQuestions = async (req: Request, res: Response): Promise<void> => {
  const questions = await prisma.commonQuestion.findMany({
    include: { category: true },
  });

  res.json(questions);
};

// 카테고리별 질문 조회
export const getQuestionsByCategory = async (req: Request, res: Response): Promise<void> => {
  const { category } = req.params;

  const foundCategory = await prisma.commonCategory.findUnique({ where: { category } });

  if (!foundCategory) {
    res.status(404).json({ error: '카테고리 없음' });
    return;
  }

  const questions = await prisma.commonQuestion.findMany({
    where: { categoryId: foundCategory.id },
  });

  res.json(questions);
};
