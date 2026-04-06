FROM python:3.10-slim

WORKDIR /app

# Copy dependencies
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose API port
EXPOSE 8000

# Run API server
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
