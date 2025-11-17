FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    JUPYTER_ENABLE_LAB=yes
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data/ ./data/
COPY src/ ./src/
COPY readme.md .

# Jupyter Port
EXPOSE 8888

# Create a startup script
RUN echo '#!/bin/bash\njupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token="" --NotebookApp.password=""' > /app/start.sh && \
    chmod +x /app/start.sh

# Run Jupyter notebook
CMD ["/app/start.sh"]
