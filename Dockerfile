# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project (simplifies path handling)
COPY . .

# Ensure model directory exists (safe guard)
RUN mkdir -p /app/models

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit app (adjust path to match structure)
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]