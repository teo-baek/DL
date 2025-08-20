FROM python:3.12-slim

WORKDIR /app

COPY requirments.txt .

RUN pip install -r requirments.txt
RUN pip install uvicorn fastapi
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

COPY 실습폴더/infermain.py /app/infermain.py

COPY model/ /app/model/
COPY imgdata/ /app/imgdata/

EXPOSE 8400
CMD [ "uvicorn", "infermain:app", "--host", "0.0.0.0", "--port", "8400" ]

