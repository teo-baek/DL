from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def video():
    return {'result': '반갑습니다.'}

@app.get('/video')
def video():
    return {'result': '동영상 생성이 완료되었습니다.'}