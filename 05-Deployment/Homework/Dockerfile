FROM agrigorev/zoomcamp-model:2025

# Use the same working directory as the base image
WORKDIR /code

# Copy dependency file and FastAPI script
COPY pyproject.toml .
COPY predict.py .

# Install Python dependencies
RUN pip install --no-cache-dir fastapi uvicorn pydantic scikit-learn

# Expose the port
EXPOSE 9697

# Run the application with uvicorn
ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9697"]