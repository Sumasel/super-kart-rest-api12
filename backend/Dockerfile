# Use a minimal base image with Python 3.9 installed
FROM python:3.9-slim

# Set the working directory inside the container to /app
WORKDIR /app

# Copy the saved model and preprocessor
COPY best_random_forest_model.pkl best_random_forest_model.pkl
COPY preprocessor.pkl preprocessor.pkl

# Copy the application code
COPY app.py app.py

# Install Python dependencies listed in requirements.txt
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:7860", "app:app"]

