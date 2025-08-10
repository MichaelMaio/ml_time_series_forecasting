FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy all pipeline scripts and wrapper
COPY train.py .
COPY promote.py .
COPY inference.py .
COPY prophet_wrapper.py .

# Default command (can be overridden by docker-compose)
CMD ["python", "train.py"]