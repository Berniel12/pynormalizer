FROM python:3.10-slim

# Set working directory
WORKDIR /usr/src/app

# Copy requirements file
COPY pyproject.toml .

# Install dependencies
RUN pip install --no-cache-dir pip==24.0 setuptools==69.1.0 wheel==0.42.0
RUN pip install --no-cache-dir pydantic-ai==0.0.31
RUN pip install --no-cache-dir deep-translator python-dateutil
RUN pip install --no-cache-dir .

# Copy project files
COPY . .

# Set up entrypoint
CMD ["python", "-m", "src.main"] 