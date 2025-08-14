FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY src/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy all pipeline scripts
COPY src/ ./src
