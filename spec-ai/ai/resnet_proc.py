import errno
import os
import torch
import torch.nn as nn
import cv2
import numpy as np
import torchvision.transforms as T
from typing import Tuple
from ai.models.resnet9_model import ResNet9  # 이미 정의된 모델을 임포트
from loguru import logger

EMOTIONS = {0: "Negative", 1: "Neutral", 2: "Positive"}

MODELS = {
    "ResNet9": "ai/models/ResNet9/ResNet9_epoch-198_score-0.846.pth",  # 모델 파일 경로
}

class ResNet9Handler:
    def __init__(self) -> None:
        # 얼굴 인식을 위한 classifier 정의
        self._classifier = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        # 디바이스 설정 (GPU가 있으면 GPU 사용)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._load_model()

    def _load_model(self) -> ResNet9:
        model_path = os.path.join(os.getcwd(), MODELS["ResNet9"])

        if not os.path.exists(model_path):
            msg = f"No model file found at {model_path}"
            logger.error(msg)
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), model_path)

        model = ResNet9(in_channels=3, num_classes=3)  # num_classes=3으로 설정

        # 가중치를 불러오기
        state_dict = torch.load(model_path, map_location=self._device)
        model_state_dict = model.state_dict()

        # 사이즈가 맞지 않는 가중치 제외하고 로드
        for name, param in state_dict.items():
            if name in model_state_dict and param.shape == model_state_dict[name].shape:
                model_state_dict[name].copy_(param)
            else:
                logger.warning(f"Skipping loading parameter {name} due to size mismatch.")

        model.load_state_dict(model_state_dict)
        model.eval()  # 평가 모드로 설정
        return model


    def get_model(self) -> ResNet9:
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _to_device(self, data):
        if isinstance(data, (list, tuple)):
            return [self._to_device(x) for x in data]
        return data.to(self._device, non_blocking=True)

    def detect_faces(self, img: np.ndarray, dsize: Tuple[int] = (224, 224)):
        if img is None:
            raise ValueError("img is required")

        # 얼굴 인식
        faces = self._classifier.detectMultiScale(
            img, scaleFactor=1.1, minNeighbors=5, minSize=dsize
        )
        if len(faces) == 0:
            return None, None

        face_img = None

        for x, y, w, h in faces:
            m = max(w, h)
            cv2.rectangle(img, (x, y), (x + m, y + m), (0, 255, 0), 2)
            face_img = img[y : y + m, x : x + m].copy()
            if dsize is not None:
                face_img = cv2.resize(face_img, dsize=dsize)

        return img, face_img

    def save_thumbnail(self, img: np.ndarray, path: str):
        try:
            logger.debug(f"Save thumbnail to {path}")

            # 썸네일 크기 조정
            width = 540
            height = int(img.shape[0] * (width / img.shape[1]))
            thumbnail = cv2.resize(img, dsize=(width, height))
            logger.debug(
                f"Resize thumbnail from ({img.shape[1]}, {img.shape[0]}) to ({width}, {height})"
            )

            thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
            cv2.imwrite(path, thumbnail)
        except Exception as e:
            logger.error("Failed to create a thumbnail: {}", e)

    def predict(self, img: np.ndarray, dsize: Tuple[int] = (224, 224)):
        # 이미지 전처리: BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR을 RGB로 변환
        transform = T.Compose([T.Resize(dsize), T.ToTensor()])
        img = transform(img)

        # 모델에 입력하기 전에 장치로 데이터를 이동
        x = self._to_device(img.unsqueeze(0))

        # 모델 예측
        with torch.no_grad():
            y = self._model(x)

        _, preds = torch.max(y, dim=1)

        return EMOTIONS[preds[0].item()]

# 사용 예시
resnet9_handler = ResNet9Handler()
