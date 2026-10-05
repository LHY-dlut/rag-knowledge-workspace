FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml requirements.lock /app/
ARG INSTALL_OCR=false
RUN pip install --no-cache-dir --require-hashes -r requirements.lock
COPY app /app/app
RUN pip install --no-cache-dir --no-deps . && if [ "$INSTALL_OCR" = "true" ]; then pip install --no-cache-dir ".[ocr]"; fi && useradd --create-home rag && mkdir -p /app/data/uploads && chown -R rag:rag /app
USER rag
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
