FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential libgl1 libglib2.0-0 tesseract-ocr && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt && python -m spacy download en_core_web_sm
COPY . /app
CMD ["python", "-m", "src.cli", "--help"]
