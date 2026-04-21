FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY config.py .
COPY models.py .
COPY filters.py .
COPY retrieval.py .
COPY llm_generation.py .
COPY utils.py .
COPY pipeline.py .
COPY main.py .
COPY example_query.json .
COPY api.py .
# Create outputs directory
RUN mkdir -p outputs

# Set environment variables for the application
# These will be overridden by docker-compose or runtime environment
ENV PINECONE_API_KEY=""

# Run the application
EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "300", "api:app"]
