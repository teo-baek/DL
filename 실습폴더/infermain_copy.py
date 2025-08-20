from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel # 데이터 예외처리시 사용
from typing import List # 타입 체크시 사용
from PIL import Image # 이미지 처리시 사용
import torch # 모델 사용시 사용
import torch.nn as nn 
import torchvision.transforms as transforms # 이미지 처리시 사용
import torchvision.models as models # 모델 사용시 사용
import json

app = FastAPI(title= 'ResNet34 Inference')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = models.resnet34(pretrained=True)
model.fc = nn.Linear(in_features= 512, out_features=3, bias=True) # 기존 코드에서 그대로 붙여넣기
model.load_state_dict(torch.load('C:/DL/model/mymodel.pth'))
model.eval() # 추론모델 함수 기능 정지
model.to(device)# 모델을 gpu에 올려야 함
# 모델을 디바이스에 올려야 함

transforms_infer = transforms.Compose(
    [ # 원래는 사람의 얼굴만 잘라오는 라이브러리를 사용함 이렇게 가로세로 다른 이미지를 224로 통일하면 깨짐
        transforms.Resize((224,224)), # 이미지넷이 이 규격임 # 이중괄호호
        transforms.ToTensor(), # 텐서로 변환
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)) # 정규화
    ]
)
# 이미지 전처리 코드 그대로 붙여넣기

# 상대방에게 전달할시 데이터 타입 정의
class response(BaseModel):
    name: str
    score: float
    type: int

@app.post("/predict", response_model=response) #response_model=response 응답시 타입 정의
async def predict(file: UploadFile = File(...), file1: UploadFile = File(...)): # endpoint로 들어올때 파일 필요, file: 은 키값 수정해서 이 타입으로 , 
    # UploadFile = File(...) 은 fastapi가 정해놓은 형태
    # 이미지 2개 보낼때를 가정
    image = Image.open(file.file) # 이미지 불러옴
    image.save('./imgdata/test.jpg') #실제로는 test를를 uuid 카운트, timestamp 로 저장
    img_tensor = transforms_infer(image).unsqueeze(0).to(device) # 이미지 전처리 
    # unsqueeze(0) 삼차원을 [1,3,224,224]으로 배치 추가 위해 차원 추가
    # to(device) 이미지를 디바이스에 올려야 함

    with torch.no_grad():
        pred = model(img_tensor)
        print('예측값:', pred)
    pred_result = torch.max(pred, dim=1)[1].item() # 최대값 인덱스 추출 # 0,1,2 # [1]은 모델이 정답으로 예측한 값값
    score = nn.Softmax()(pred)[0] # softmax 함수 적용 3개의 값을 1로 봤을때 90% 값으로 카리나로 예측 # [0.03,0.09,0.07]
    # [0]은 모델의 예측값들 퍼센테이지 # [1]은 해당 점수들의 인덱스들들
    # [0.9,0.3]->예측값들들, [0,1] ->인덱스[1]해당값의 인덱스 
    print('Softmax 적용 후 값:', score)
    classname = ['마동석','카리나','차은우']
    name = classname[pred_result] # 결과 값 반영해서 이름
    print('name:', name)
    return response(name=name, score=float(score[pred_result]), type=pred_result) # return 은 요청한 상대한테
# @app.post("/predict", response_model=response) #response_model=response 응답시 타입 정의
# async def predict(file: UploadFile = File(...)): # endpoint로 들어올때 파일 필요, file: 은 키값 수정해서 이 타입으로 , 
#     return response(name='test1', score=0.123, type=1) #이런식으로 더미 데이터 넣어서 미리 만들도록 줘야 함
