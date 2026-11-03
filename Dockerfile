FROM python:3.10-slim
WORKDIR /opt/qgvi
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["python", "-m", "runtime.cli", "demo", "--no-browser", "--bind", "0.0.0.0"]
