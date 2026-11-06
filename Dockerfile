FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

COPY requirements-docker.txt requirements-docker.txt
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY . .

CMD ["python", "-m", "src.main"]
