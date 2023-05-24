# lightweight Python image 
FROM python:3.9-slim
LABEL maintainer="szecsinora2000@gmail.com"

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory to /app
WORKDIR /app

RUN python -m venv /myenv
ENV PATH="/myenv/bin:$PATH"

#Install git so that we can clone the app code from a remote repo:
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    r-base \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip

# Clone your code that lives in a remote repo to WORKDIR
# RUN git clone https://github.com/streamlit/streamlit-example.git .

# Install any dependencies specified in requirements.txt
RUN pip install numpy
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# vagy helyette ha If your code lives in the same directory as the Dockerfile, copy all your app files from your server into the container
COPY . .

# Expose port 8501 for Streamlit
EXPOSE 8501

# The HEALTHCHECK instruction tells Docker how to test a container to check that it is still working. Your container needs to listen to Streamlit’s (default) port 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Start Streamlit when the container launches
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.maxUploadSize=1028"]  
# "--server.port=8501", "--server.address=0.0.0.0",
