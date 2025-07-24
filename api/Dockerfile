# Use official Python base image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy everything into container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Expose the FastAPI port
EXPOSE 8000

# Run the app
CMD ["uvicorn", "api.predict:app", "--host", "0.0.0.0", "--port", "8000"]
