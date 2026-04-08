# FROM python:3.10

# WORKDIR /app

# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# COPY . .

# # Expose the port FastAPI runs on
# EXPOSE 7860

# # Run the infeFrence API
# CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "7860"]

FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (much smaller)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Then install rest
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]