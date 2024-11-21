# Use a smaller base image (like python:3.10-slim)
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the application files (including house_price_api.py) into the container
COPY house_price_api.py /app/
COPY train.csv /app/train.csv
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app will run on
EXPOSE 8000

# Run the app using uvicorn (make sure the app path matches your file name)
CMD ["uvicorn", "house_price_api:app", "--host", "0.0.0.0", "--port", "8000"]
