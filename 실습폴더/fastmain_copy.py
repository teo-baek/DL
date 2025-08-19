from fastapi import FastAPI

app = FastAPI()

@app.get('/video')
def video():
    return {'result': '동영상 생성이 완료되었습니다.'}

@app.get('/chatbot')
def chatbot():
    return {'result': '안녕하세요'}