FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
WORKDIR /code
COPY ./requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt
COPY ./Models ./Models
COPY ./server.py ./server.py
EXPOSE 8000

CMD ["python","-m","uvicorn","server:app", "--host" , "0.0.0.0" , "--port" , "8000"]
