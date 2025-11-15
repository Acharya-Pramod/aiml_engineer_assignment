# Use official lightweight Python image
FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app code
COPY app /app/app

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

# Use uvicorn to run the FastAPI app; Cloud Run expects port 8080 commonly
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

