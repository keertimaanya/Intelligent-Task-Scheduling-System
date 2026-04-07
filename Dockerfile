FROM python:3.10-slim

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy and install dependencies
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY --chown=user . /app

# HF Spaces uses port 7860
EXPOSE 7860

# Set default port for HF Spaces compatibility
ENV PORT=7860

# Run API server on the correct port
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "7860"]
