# Base stage for system dependencies
FROM mcr.microsoft.com/devcontainers/python:1-3.11-bullseye as base
WORKDIR /app
COPY packages.txt .
RUN apt-get update && \
    apt-get install -y --no-install-recommends $(cat packages.txt) && \
    rm -rf /var/lib/apt/lists/*

# Dependencies stage for Python packages
FROM base as dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Final stage for the application
FROM dependencies as final
COPY . /app
EXPOSE 8501
CMD ["streamlit", "run", "Hello.py", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
