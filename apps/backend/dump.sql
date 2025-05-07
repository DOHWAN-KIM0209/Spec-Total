-- MySQL dump 10.13  Distrib 8.0.41, for Linux (x86_64)
--
-- Host: localhost    Database: interview_ai
-- ------------------------------------------------------
-- Server version	8.0.41-0ubuntu0.24.04.1

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `Analysis`
--

DROP TABLE IF EXISTS `Analysis`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Analysis` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `userId` bigint NOT NULL,
  `type` varchar(191) COLLATE utf8mb4_unicode_ci NOT NULL,
  `question` varchar(191) COLLATE utf8mb4_unicode_ci NOT NULL,
  `videoPath` varchar(191) COLLATE utf8mb4_unicode_ci NOT NULL,
  `thumbnailPath` varchar(191) COLLATE utf8mb4_unicode_ci NOT NULL,
  `keyword` varchar(191) COLLATE utf8mb4_unicode_ci NOT NULL,
  `setStartTime` datetime(3) NOT NULL,
  `analysisReqTime` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  `analysisStartTime` datetime(3) NOT NULL,
  `analysisEndTime` datetime(3) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `Analysis_userId_fkey` (`userId`),
  CONSTRAINT `Analysis_userId_fkey` FOREIGN KEY (`userId`) REFERENCES `User` (`id`) ON DELETE RESTRICT ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Analysis`
--

LOCK TABLES `Analysis` WRITE;
/*!40000 ALTER TABLE `Analysis` DISABLE KEYS */;
/*!40000 ALTER TABLE `Analysis` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `CommonCategory`
--

DROP TABLE IF EXISTS `CommonCategory`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `CommonCategory` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `category` varchar(191) COLLATE utf8mb4_unicode_ci NOT NULL,
  `createdTime` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  `updatedTime` datetime(3) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `CommonCategory_category_key` (`category`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `CommonCategory`
--

LOCK TABLES `CommonCategory` WRITE;
/*!40000 ALTER TABLE `CommonCategory` DISABLE KEYS */;
/*!40000 ALTER TABLE `CommonCategory` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `CommonKeyword`
--

DROP TABLE IF EXISTS `CommonKeyword`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `CommonKeyword` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `userId` bigint NOT NULL,
  `questionId` bigint NOT NULL,
  `keyword` varchar(191) COLLATE utf8mb4_unicode_ci NOT NULL,
  `createdTime` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  `updatedTime` datetime(3) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `CommonKeyword_userId_fkey` (`userId`),
  KEY `CommonKeyword_questionId_fkey` (`questionId`),
  CONSTRAINT `CommonKeyword_questionId_fkey` FOREIGN KEY (`questionId`) REFERENCES `CommonQuestion` (`id`) ON DELETE RESTRICT ON UPDATE CASCADE,
  CONSTRAINT `CommonKeyword_userId_fkey` FOREIGN KEY (`userId`) REFERENCES `User` (`id`) ON DELETE RESTRICT ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `CommonKeyword`
--

LOCK TABLES `CommonKeyword` WRITE;
/*!40000 ALTER TABLE `CommonKeyword` DISABLE KEYS */;
/*!40000 ALTER TABLE `CommonKeyword` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `CommonQuestion`
--

DROP TABLE IF EXISTS `CommonQuestion`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `CommonQuestion` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `categoryId` bigint NOT NULL,
  `question` varchar(191) COLLATE utf8mb4_unicode_ci NOT NULL,
  `createdTime` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  `updatedTime` datetime(3) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `CommonQuestion_categoryId_fkey` (`categoryId`),
  CONSTRAINT `CommonQuestion_categoryId_fkey` FOREIGN KEY (`categoryId`) REFERENCES `CommonCategory` (`id`) ON DELETE RESTRICT ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `CommonQuestion`
--

LOCK TABLES `CommonQuestion` WRITE;
/*!40000 ALTER TABLE `CommonQuestion` DISABLE KEYS */;
/*!40000 ALTER TABLE `CommonQuestion` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `CommonScript`
--

DROP TABLE IF EXISTS `CommonScript`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `CommonScript` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `userId` bigint NOT NULL,
  `questionId` bigint NOT NULL,
  `script` varchar(191) COLLATE utf8mb4_unicode_ci NOT NULL,
  `createdTime` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  `updatedTime` datetime(3) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `CommonScript_userId_fkey` (`userId`),
  KEY `CommonScript_questionId_fkey` (`questionId`),
  CONSTRAINT `CommonScript_questionId_fkey` FOREIGN KEY (`questionId`) REFERENCES `CommonQuestion` (`id`) ON DELETE RESTRICT ON UPDATE CASCADE,
  CONSTRAINT `CommonScript_userId_fkey` FOREIGN KEY (`userId`) REFERENCES `User` (`id`) ON DELETE RESTRICT ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `CommonScript`
--

LOCK TABLES `CommonScript` WRITE;
/*!40000 ALTER TABLE `CommonScript` DISABLE KEYS */;
/*!40000 ALTER TABLE `CommonScript` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `Resume`
--

DROP TABLE IF EXISTS `Resume`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Resume` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `userId` bigint NOT NULL,
  `name` varchar(191) COLLATE utf8mb4_unicode_ci NOT NULL,
  `filePath` varchar(191) COLLATE utf8mb4_unicode_ci NOT NULL,
  `createdTime` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  `updatedTime` datetime(3) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `Resume_userId_fkey` (`userId`),
  CONSTRAINT `Resume_userId_fkey` FOREIGN KEY (`userId`) REFERENCES `User` (`id`) ON DELETE RESTRICT ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Resume`
--

LOCK TABLES `Resume` WRITE;
/*!40000 ALTER TABLE `Resume` DISABLE KEYS */;
/*!40000 ALTER TABLE `Resume` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `ResumeKeyword`
--

DROP TABLE IF EXISTS `ResumeKeyword`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `ResumeKeyword` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `userId` bigint NOT NULL,
  `questionId` bigint NOT NULL,
  `resumeId` bigint NOT NULL,
  `keyword` varchar(191) COLLATE utf8mb4_unicode_ci NOT NULL,
  `createdTime` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  `updatedTime` datetime(3) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `ResumeKeyword_userId_fkey` (`userId`),
  KEY `ResumeKeywordToResume` (`resumeId`),
  CONSTRAINT `ResumeKeyword_userId_fkey` FOREIGN KEY (`userId`) REFERENCES `User` (`id`) ON DELETE RESTRICT ON UPDATE CASCADE,
  CONSTRAINT `ResumeKeywordToResume` FOREIGN KEY (`resumeId`) REFERENCES `Resume` (`id`) ON DELETE RESTRICT ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `ResumeKeyword`
--

LOCK TABLES `ResumeKeyword` WRITE;
/*!40000 ALTER TABLE `ResumeKeyword` DISABLE KEYS */;
/*!40000 ALTER TABLE `ResumeKeyword` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `ResumeQuestion`
--

DROP TABLE IF EXISTS `ResumeQuestion`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `ResumeQuestion` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `resumeId` bigint NOT NULL,
  `question` varchar(191) COLLATE utf8mb4_unicode_ci NOT NULL,
  `createdTime` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  `updatedTime` datetime(3) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `ResumeQuestion_resumeId_fkey` (`resumeId`),
  CONSTRAINT `ResumeQuestion_resumeId_fkey` FOREIGN KEY (`resumeId`) REFERENCES `User` (`id`) ON DELETE RESTRICT ON UPDATE CASCADE,
  CONSTRAINT `ResumeQuestionToResume` FOREIGN KEY (`resumeId`) REFERENCES `Resume` (`id`) ON DELETE RESTRICT ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `ResumeQuestion`
--

LOCK TABLES `ResumeQuestion` WRITE;
/*!40000 ALTER TABLE `ResumeQuestion` DISABLE KEYS */;
/*!40000 ALTER TABLE `ResumeQuestion` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `ResumeScript`
--

DROP TABLE IF EXISTS `ResumeScript`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `ResumeScript` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `userId` bigint NOT NULL,
  `questionId` bigint NOT NULL,
  `resumeId` bigint NOT NULL,
  `script` varchar(191) COLLATE utf8mb4_unicode_ci NOT NULL,
  `createdTime` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  `updatedTime` datetime(3) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `ResumeScript_userId_fkey` (`userId`),
  KEY `ResumeScriptToResume` (`resumeId`),
  CONSTRAINT `ResumeScript_userId_fkey` FOREIGN KEY (`userId`) REFERENCES `User` (`id`) ON DELETE RESTRICT ON UPDATE CASCADE,
  CONSTRAINT `ResumeScriptToResume` FOREIGN KEY (`resumeId`) REFERENCES `Resume` (`id`) ON DELETE RESTRICT ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `ResumeScript`
--

LOCK TABLES `ResumeScript` WRITE;
/*!40000 ALTER TABLE `ResumeScript` DISABLE KEYS */;
/*!40000 ALTER TABLE `ResumeScript` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `Social`
--

DROP TABLE IF EXISTS `Social`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Social` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `provider` varchar(191) COLLATE utf8mb4_unicode_ci NOT NULL,
  `providerId` bigint NOT NULL,
  `userId` bigint NOT NULL,
  PRIMARY KEY (`id`),
  KEY `Social_userId_fkey` (`userId`),
  CONSTRAINT `Social_userId_fkey` FOREIGN KEY (`userId`) REFERENCES `User` (`id`) ON DELETE RESTRICT ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Social`
--

LOCK TABLES `Social` WRITE;
/*!40000 ALTER TABLE `Social` DISABLE KEYS */;
/*!40000 ALTER TABLE `Social` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `User`
--

DROP TABLE IF EXISTS `User`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `User` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `role` varchar(191) COLLATE utf8mb4_unicode_ci NOT NULL,
  `email` varchar(191) COLLATE utf8mb4_unicode_ci NOT NULL,
  `password` varchar(191) COLLATE utf8mb4_unicode_ci NOT NULL,
  `name` varchar(191) COLLATE utf8mb4_unicode_ci NOT NULL,
  `profileImage` varchar(191) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `deletedTime` datetime(3) DEFAULT NULL,
  `createdTime` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  `updatedTime` datetime(3) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `User_email_key` (`email`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `User`
--

LOCK TABLES `User` WRITE;
/*!40000 ALTER TABLE `User` DISABLE KEYS */;
/*!40000 ALTER TABLE `User` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `_prisma_migrations`
--

DROP TABLE IF EXISTS `_prisma_migrations`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `_prisma_migrations` (
  `id` varchar(36) COLLATE utf8mb4_unicode_ci NOT NULL,
  `checksum` varchar(64) COLLATE utf8mb4_unicode_ci NOT NULL,
  `finished_at` datetime(3) DEFAULT NULL,
  `migration_name` varchar(255) COLLATE utf8mb4_unicode_ci NOT NULL,
  `logs` text COLLATE utf8mb4_unicode_ci,
  `rolled_back_at` datetime(3) DEFAULT NULL,
  `started_at` datetime(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  `applied_steps_count` int unsigned NOT NULL DEFAULT '0',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `_prisma_migrations`
--

LOCK TABLES `_prisma_migrations` WRITE;
/*!40000 ALTER TABLE `_prisma_migrations` DISABLE KEYS */;
INSERT INTO `_prisma_migrations` VALUES ('478ca7b9-5b27-47a6-bace-3c42e5c4712c','1ac7db420f964a8e0d0472943a8d6226fca7c0038cc0d80b8fff57940bc2729d','2025-04-16 06:35:05.763','20250416063503_init',NULL,NULL,'2025-04-16 06:35:03.842',1);
/*!40000 ALTER TABLE `_prisma_migrations` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2025-05-02 22:08:19
