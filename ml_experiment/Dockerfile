FROM python:3.10

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY palmer-panguin-decisionTree.py .

CMD ["python", "palmer-panguin-decisionTree.py"]