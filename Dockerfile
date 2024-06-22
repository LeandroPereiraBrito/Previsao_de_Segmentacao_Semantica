# Use the official Python image as a base
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install build essentials (including compilers), git, and any other necessary dependencies
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
        build-essential \
        cmake \
        libjpeg-dev \
        zlib1g-dev \
        git \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . .

# Expose the port on which the FastAPI application will run
EXPOSE 5000

# Command to run the application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]