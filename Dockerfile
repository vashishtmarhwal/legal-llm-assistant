# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    POETRY_VERSION=1.5.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    PYTHONPATH="/app"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy project dependency files
COPY pyproject.toml poetry.lock* ./

# Install project dependencies
RUN poetry install --no-dev --no-root

# Copy the rest of the application
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p /app/data /app/logs \
    && chmod -R 755 /app/data /app/logs

# Create a non-root user and switch to it
RUN useradd -m appuser \
    && chown -R appuser:appuser /app
USER appuser

# Expose the port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]