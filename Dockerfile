FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10
WORKDIR /app
COPY ./app /app
COPY requirements.txt /app
ENV PYTHONPATH=/app
RUN pip install --no-cache-dir -r requirements.txt --extra -index-url https://download.pytorch.org/whl/cpu
# Debug: List all files and directories in /app
RUN find /app -type f -o -type d
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]