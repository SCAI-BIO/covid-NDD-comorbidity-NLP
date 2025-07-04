# Multi-stage Dockerfile for COVID-NDD Comorbidity Analysis Pipeline
# Stage 1: Base environment with dependencies
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Application stage
FROM base as application

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/hashes logs results

# Copy entrypoint script and make executable
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set proper permissions
RUN chmod -R 755 scripts/ cypher/

# Install application in development mode
RUN pip install -e .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python scripts/health_check.py || exit 1

# Default command
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "main.py"]

# Stage 3: Production stage (minimal)
FROM application as production

# Remove development dependencies and clean up
RUN pip uninstall -y pytest pytest-cov black flake8 \
    && apt-get autoremove -y build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Metadata
LABEL maintainer="negin.babaiha@scai.fraunhofer.de" \
      version="1.0.0" \
      description="COVID-NDD Comorbidity Analysis Pipeline" \
      org.opencontainers.image.source="https://github.com/SCAI-BIO/covid-NDD-comorbidity-NLP"