FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure the results/ folder exists even before the app creates it,
# so a bind-mounted volume (see docker-compose.yml) has somewhere to attach.
RUN mkdir -p results

# Run the benchmark when the container starts
CMD ["python", "main.py"]