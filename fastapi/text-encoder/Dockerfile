FROM python:3.8

WORKDIR /app

# install dependencies
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# copy fastAPI app codebase
COPY ./ /app

# run the fastAPI app
CMD ["uvicorn", "main:app", "--workers", "--host", "0.0.0.0", "--port", "80"]