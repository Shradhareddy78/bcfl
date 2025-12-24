FROM python:3.8-slim

RUN pip install sawtooth-sdk

WORKDIR /app
COPY tp_simple.py .

CMD ["python", "tp_simple.py"]
