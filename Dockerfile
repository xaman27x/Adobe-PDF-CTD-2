# Dockerfile
FROM python:3.9-slim as builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY setup.py .
RUN python setup.py

FROM python:3.9-slim

ENV PLATFORM=linux/amd64

WORKDIR /app

RUN pip install --no-cache-dir numpy onnxruntime yake PyMuPDF

COPY src/ /app/src/

COPY --from=builder /app/models /app/models

VOLUME /app/input
VOLUME /app/output

CMD ["python", "-m", "src.solution"]