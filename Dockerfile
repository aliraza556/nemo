# Use an official Python runtime as a parent image
FROM python:3.11

ENV PIP_DEFAULT_TIMEOUT=100 \
    # Allow statements and log messages to immediately appear
    PYTHONUNBUFFERED=1 \
    # disable a pip version check to reduce run-time & log-spam
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # cache is useless in docker image, so disable to reduce image size
    PIP_NO_CACHE_DIR=1

# Set the working directory to /app
WORKDIR /app

COPY ./requirements.txt /app/requirements.txt
# Install any needed packages specified in requirements.txt
RUN apt-get update && apt-get install build-essential -y && apt-get install -y --no-install-recommends git ffmpeg && \
    set -ex apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

RUN pip install Cython
RUN pip install megatron-core==0.3.0
RUN pip install git+https://github.com/NVIDIA/NeMo.git@r1.22.0#egg=nemo_toolkit[asr]
# RUN pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[all]
RUN pip install -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Define environment variable
ENV INPUT_FILE_URL='https://whinx-uploads.s3.amazonaws.com/Demucs/18eb998c-3015-478b-a6d5-01d147519823.wav'

# Run app.py when the container launches
CMD ["python", "app.py"]