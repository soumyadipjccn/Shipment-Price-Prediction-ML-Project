FROM python:3.11-slim-buster

WORKDIR /app

# Install system dependencies & AWS CLI dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt setup.py ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . /app

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
