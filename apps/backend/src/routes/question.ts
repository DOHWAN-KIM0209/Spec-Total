import express from 'express';
import {  createQuestion,  getAllQuestions,  getQuestionsByCategory} from '../controllers/questionController';

const router = express.Router();

router.post('/', createQuestion);
router.get('/', getAllQuestions);
router.get('/:category', getQuestionsByCategory);

export default router;
