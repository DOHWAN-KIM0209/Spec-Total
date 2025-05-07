import numpy as np
import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import BertModel, BertTokenizer  # KoBERT 관련 import

# 캐시 위치 설정 (옵션)
os.environ["HF_HOME"] = "C:/Users/project1/huggingface_cache"
os.environ['TRANSFORMERS_CACHE'] = 'C:\\Users\\project1'

# 문장 변환 클래스
class BERTSentenceTransform:
    def __init__(self, tokenizer, max_seq_length, pad=True, pair=False):
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length
        self._pad = pad
        self._pair = pair

    def __call__(self, line):
        text_a = line[0]
        text_b = line[1] if self._pair else None

        encoding = self._tokenizer.encode_plus(
            text_a,
            text_b,
            max_length=self._max_seq_length,
            padding="max_length" if self._pad else False,
            truncation=True,
            return_token_type_ids=True,
        )

        return (
            np.array(encoding["input_ids"], dtype="int32"),
            np.array(sum(1 for id in encoding["attention_mask"] if id == 1), dtype="int32"),
            np.array(encoding["token_type_ids"], dtype="int32"),
        )

# 데이터셋 클래스
class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, tokenizer, max_len, pad, pair):
        transform = BERTSentenceTransform(
            tokenizer, max_seq_length=max_len, pad=pad, pair=pair
        )
        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return self.sentences[i] + (self.labels[i],)

    def __len__(self):
        return len(self.labels)

# 분류기 클래스
class BERTClassifier(nn.Module):
    def __init__(self, bert, hidden_size=768, num_classes=52, dr_rate=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        outputs = self.bert(
            input_ids=token_ids,
            token_type_ids=segment_ids,
            attention_mask=attention_mask.to(token_ids.device),
        )
        pooler = outputs[0][:, 0]  # CLS 토큰
        out = self.dropout(pooler) if self.dr_rate else pooler
        return self.classifier(out)

# 래퍼 클래스
class KoBERTModelWrapper:
    def __init__(self) -> None:
        self._pretrained = "monologg/kobert"

        self._tokenizer = BertTokenizer.from_pretrained(self._pretrained)
        self._bert_model = BertModel.from_pretrained(self._pretrained)
        self._model = BERTClassifier(self._bert_model, num_classes=52)
        self._device = torch.device("cpu")

        # KoBERT는 학습된 state_dict가 없을 수도 있으니 이 부분은 주석 처리하거나 에러 방지용 예외 처리
        model_path = os.path.join(os.getcwd(), "ai/models/KoBERT/model_state_dict.pt")
        if os.path.exists(model_path):
            self._model.load_state_dict(torch.load(model_path, map_location=self._device))
        else:
            print("⚠️ 사전 학습된 모델 파라미터가 없습니다. 기본 KoBERT로 실행합니다.")

        self._model.to(self._device)

        self._intent_labels = pd.read_csv(
            os.path.join(os.getcwd(), "ai/data/intent_labels.tsv"),
            sep="\t",
            encoding="utf8",
        )

        self._max_len = 128
        self._batch_size = 64

    def get_intent_labels(self) -> pd.DataFrame:
        return self._intent_labels

    def predict(self, sentence):
        dataset = [[sentence, "0"]]
        test = BERTDataset(
            dataset, 0, 1, self._tokenizer, self._max_len, True, False
        )
        test_dataloader = torch.utils.data.DataLoader(
            test, batch_size=self._batch_size, num_workers=0
        )

        self._model.eval()
        answer = None

        for _, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
            token_ids = token_ids.long().to(self._device)
            segment_ids = segment_ids.long().to(self._device)
            out = self._model(token_ids, valid_length, segment_ids)
            for logits in out:
                logits = logits.detach().cpu().numpy()
                answer = sorted(
                    {i: v for i, v in enumerate(logits) if v > 0}.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:5]
        return answer


# 모델 인스턴스화
kobert_model = KoBERTModelWrapper()