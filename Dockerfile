# Use Python 3.11 base image (slim version for smaller size)
FROM python:3.11-slim

# Set environment variables to prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Set working directory
WORKDIR /app

# Install basic system packages needed for C-extensions (like faiss or spacy parsing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker build cache
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download spaCy NLP model required for extraction
RUN python -m spacy download en_core_web_sm

# Copy the rest of the application files
COPY . .

# Expose port 8501 for Streamlit access
EXPOSE 8501

# Set the entrypoint to run the Streamlit app
ENTRYPOINT ["streamlit", "run", "app.py"]
