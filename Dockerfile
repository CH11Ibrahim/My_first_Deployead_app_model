# Use Miniconda as the base image
FROM continuumio/miniconda3

# Set the working directory
WORKDIR /app

# Install Python 3.12.7 and necessary dependencies
RUN conda install -y python=3.12.7 && \
    conda install -y pip && \
    apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone your repository
RUN git clone https://github.com/CH11Ibrahim/My_first_Deployed_Model.git .

RUN pip install --upgrade pip setuptools

# Install required Python packages directly
RUN pip install --no-cache-dir streamlit pandas joblib numpy setuptools matplotlib seaborn scikit-learn xgboost

# Expose Streamlit's default port
EXPOSE 8501

# Add health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Set entry point for running Streamlit
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
