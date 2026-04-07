FROM python:3.10-slim

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# HF Spaces uses port 7860
EXPOSE 7860

# Set default port for HF Spaces compatibility
ENV PORT=7860

# Run API server on the correct port
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "7860"]
