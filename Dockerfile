FROM python:3.9

WORKDIR /app

COPY . .

RUN pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install numpy gymnasium stable-baselines3 matplotlib
CMD ["python", "inference.py"]