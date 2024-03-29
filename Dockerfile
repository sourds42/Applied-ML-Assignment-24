# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY app.py score.py /app/
COPY train.csv .
COPY best_model.joblib .

# Install any needed packages specified in requirements.txt
# Ensure you have a requirements.txt file with Flask and other dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


# Make port 5000 available to the world outside this container
EXPOSE 5000
# configure the container to run in an executed manner
ENTRYPOINT [ "python" ]
CMD ["app.py", "score.py" ]
