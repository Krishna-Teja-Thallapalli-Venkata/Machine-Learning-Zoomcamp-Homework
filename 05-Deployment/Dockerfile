FROM python:3.8.12-slim
 
RUN pip install pipenv
 
WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]
 
RUN pip install flask gunicorn scikit-learn pandas numpy
 
COPY ["Predict.py", "model_C=1.0.bin", "./"]
 
EXPOSE 9696
 
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "Predict:app"]