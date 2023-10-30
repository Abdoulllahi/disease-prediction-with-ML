FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN python -m venv venv
RUN . venv/bin/activate

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 80

ENV NAME KaggleX

CMD ["python", "app/app.py"]