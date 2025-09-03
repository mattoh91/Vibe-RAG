FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 12001

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port", "12001", "--server.address", "0.0.0.0", "--server.headless", "true", "--server.fileWatcherType", "none"]