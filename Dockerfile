FROM python:3.8-slim

WORKDIR /app

# ---- FORCE PYTHON PROTOBUF IMPLEMENTATION ----
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Install deps WITHOUT protobuf first
RUN pip install --no-cache-dir flwr sawtooth-sdk requests numpy

# FORCE protobuf downgrade AFTER everything
RUN pip install --no-cache-dir --force-reinstall protobuf==3.20.3

COPY server.py .

EXPOSE 8080

CMD ["python", "server.py"]