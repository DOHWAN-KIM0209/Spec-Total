import express from 'express';
import {  uploadResume,  getUserResumes} from '../controllers/resumeController';

const router = express.Router();

router.post('/', uploadResume);
router.get('/:userId', getUserResumes);

export default router;
