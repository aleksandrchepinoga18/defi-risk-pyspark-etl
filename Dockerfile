
FROM apache/spark:3.5.5-python3

USER root

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/

CMD ["python3", "-u", "src/etl_pipeline.py"]
