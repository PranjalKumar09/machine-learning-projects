# ===============================
# Author: Pranjal Kumar Shukla
# GitHub: https://github.com/PranjalKumar09/machine-learning-projects
# ===============================

FROM python:3.10

WORKDIR /app
COPY requirements.txt .
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt
COPY src/ src/
COPY csv/ static/

CMD ["python", "static/Big_market_sales.py"]
