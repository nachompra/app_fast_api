FROM python:3.9.21

WORKDIR /app

COPY requirements.txt .
COPY app /app
COPY data /app/data

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
