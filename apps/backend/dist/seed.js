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
Object.defineProperty(exports, "__esModule", { value: true });
const client_1 = require("@prisma/client");
const prisma = new client_1.PrismaClient();
function seed() {
    return __awaiter(this, void 0, void 0, function* () {
        yield prisma.$executeRawUnsafe(`
    INSERT INTO \`_prisma_migrations\` 
    VALUES (
      '478ca7b9-5b27-47a6-bace-3c42e5c4712c',
      '1ac7db420f964a8e0d0472943a8d6226fca7c0038cc0d80b8fff57940bc2729d',
      '2025-04-16 06:35:05.763',
      '20250416063503_init',
      NULL,
      NULL,
      '2025-04-16 06:35:03.842',
      1
    );
  `);
    });
}
seed()
    .then(() => {
    console.log('✅ Seed 성공');
    process.exit(0);
})
    .catch((err) => {
    console.error('❌ Seed 실패', err);
    process.exit(1);
});
