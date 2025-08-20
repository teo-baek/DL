from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import json

app = FastAPI(title= 'ResNet34 Inference')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = models.resnet34(pretrained= True)
model.fc = nn.Linear(in_features=512, out_features=3, bias=True)
# 모델에 대한 정보는 언제나 불러올 수 있도록 저장/정리 해둬야 한다.
model.load_state_dict(torch.load('../model/mymodel.pth'))
model.eval()
model.to(device)

transforms_infer = transforms.Compose(      # 전처리를 하지 않아서 오류가 생기는 경우가 많으니 주의할 것
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)

class response(BaseModel):
    name: str
    score: float
    type: int

@app.post("/predict", response_model= response)
async def predict(file: UploadFile=File(...)):      # file은 키 값. 상대방이 나에게 보내는 키 값.
    image = Image.open(file.file)
    image.save('../dataset/test/카리나/image_7.jpg')      # 실제 현업에서는 uuid, 카운트, timestamp를 활용해서 파일명을 자동화해서 저장
    img_tensor = transforms_infer(image).unsqueeze(0).to(device)    
    # transforms_infer로 가져온 [3, 224, 224]의 이미지를 unsqueeze로 [1, 3, 224, 224] 변환해주고 GPU로 옮겨준다.
    # 형 변환하는 과정을 넣지 않아 오류가 나는 경우가 많다.

    with torch.no_grad():
        pred = model(img_tensor)
        print('예측값', pred)

    pred_result = torch.max(pred, dim= 1)[1].item()
    score = nn.Softmax()(pred)[0]      
    print('softmax', score)
    classname = ['마동석', '카리나', '이수지']
    name = classname[pred_result]
    print('name', name)

    return response(name= name, score= float(score[pred_result]), type= pred_result)
    # 키 값을 어떻게 설계할 지 기획 단계에서 충분히 고민해서 협업에게 공유해줘야 한다.
    # 키 값을 설계하는 과정에서도 협업자와 소통하면서 진행하면 좋을 듯. 
