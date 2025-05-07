"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.logout = exports.kakaoCallback = exports.kakaoLogin = exports.me = exports.login = exports.signup = void 0;
const client_1 = require("@prisma/client");
const bcrypt_1 = __importDefault(require("bcrypt"));
const jsonwebtoken_1 = __importDefault(require("jsonwebtoken"));
const axios_1 = __importDefault(require("axios"));
const prisma = new client_1.PrismaClient();
const JWT_SECRET = process.env.JWT_SECRET || 'supersecret';
const KAKAO_CLIENT_ID = process.env.KAKAO_CLIENT_ID;
const KAKAO_REDIRECT_URI = process.env.KAKAO_REDIRECT_URI;
const signup = (req, res) => __awaiter(void 0, void 0, void 0, function* () {
    const { email, password, name, role } = req.body;
    try {
        const hashedPassword = yield bcrypt_1.default.hash(password, 10);
        const user = yield prisma.user.create({
            data: {
                email,
                password: hashedPassword,
                name,
                role,
            },
        });
        res.status(201).json({ message: '회원가입 완료!', user });
    }
    catch (error) {
        res.status(400).json({ error: '이미 가입된 이메일일 수 있어요!' });
    }
});
exports.signup = signup;
const login = (req, res) => __awaiter(void 0, void 0, void 0, function* () {
    const { email, password } = req.body;
    const user = yield prisma.user.findUnique({ where: { email } });
    if (!user) {
        res.status(404).json({ error: '가입된 이메일이 아닙니다.' });
        return;
    }
    const isValid = yield bcrypt_1.default.compare(password, user.password);
    if (!isValid) {
        res.status(401).json({ error: '비밀번호가 틀렸습니다.' });
        return;
    }
    const token = jsonwebtoken_1.default.sign({ userId: user.id.toString() }, JWT_SECRET, { expiresIn: '7d' });
    res.json({ message: '로그인 성공', token });
});
exports.login = login;
const me = (req, res) => __awaiter(void 0, void 0, void 0, function* () {
    const userId = BigInt(req.userId);
    const user = yield prisma.user.findUnique({ where: { id: userId } });
    if (!user) {
        res.status(404).json({ error: '유저 정보를 찾을 수 없습니다.' });
        return;
    }
    res.json({
        user: Object.assign(Object.assign({}, user), { id: user.id.toString() }),
    });
});
exports.me = me;
const kakaoLogin = (req, res) => {
    const kakaoAuthURL = `https://kauth.kakao.com/oauth/authorize?response_type=code&client_id=${KAKAO_CLIENT_ID}&redirect_uri=${KAKAO_REDIRECT_URI}`;
    res.redirect(kakaoAuthURL);
};
exports.kakaoLogin = kakaoLogin;
const kakaoCallback = (req, res) => __awaiter(void 0, void 0, void 0, function* () {
    var _a;
    const code = req.query.code;
    const tokenRes = yield axios_1.default.post('https://kauth.kakao.com/oauth/token', null, {
        params: {
            grant_type: 'authorization_code',
            client_id: KAKAO_CLIENT_ID,
            redirect_uri: KAKAO_REDIRECT_URI,
            code,
        },
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    });
    const access_token = tokenRes.data.access_token;
    const profileRes = yield axios_1.default.get('https://kapi.kakao.com/v2/user/me', {
        headers: {
            Authorization: `Bearer ${access_token}`,
        },
    });
    const kakaoId = profileRes.data.id;
    const kakaoEmail = (_a = profileRes.data.kakao_account) === null || _a === void 0 ? void 0 : _a.email;
    let social = yield prisma.social.findFirst({
        where: {
            provider: 'kakao',
            providerId: kakaoId,
        },
        include: { user: true },
    });
    let user;
    if (social) {
        user = social.user;
    }
    else {
        user = yield prisma.user.create({
            data: {
                email: kakaoEmail || `${kakaoId}@kakao.com`,
                name: '카카오유저',
                password: '',
                role: 'user',
            },
        });
        yield prisma.social.create({
            data: {
                userId: user.id,
                provider: 'kakao',
                providerId: kakaoId,
            },
        });
    }
    const token = jsonwebtoken_1.default.sign({ userId: user.id.toString() }, JWT_SECRET, { expiresIn: '7d' });
    res.json({ message: '카카오 로그인 완료', token });
});
exports.kakaoCallback = kakaoCallback;
const logout = (req, res) => __awaiter(void 0, void 0, void 0, function* () {
    res.json({ message: '로그아웃 처리: 클라이언트에서 토큰 삭제하세요' });
});
exports.logout = logout;
