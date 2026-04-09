FROM python:3.10-slim

# System dependencies (optional, but good for science/AI projects)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Create a non-root user and set permissions
RUN useradd -m -u 1000 user
WORKDIR /app
COPY --chown=user:root . /app

# Switch to the new user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port for HF Spaces
EXPOSE 7860
ENV PORT=7860

# Run the FastAPI server
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "7860"]