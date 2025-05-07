import express from 'express';
import {  createScript,  getUserScripts} from '../controllers/scriptController';

const router = express.Router();

router.post('/', createScript);
router.get('/:userId', getUserScripts);

export default router;
