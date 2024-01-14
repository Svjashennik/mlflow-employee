FROM python:3.10.6-slim
WORKDIR /app/
COPY . .
RUN python3 -m pip install --no-cache-dir --no-warn-script-location --upgrade pip \
    && python3 -m pip install --no-cache-dir --no-warn-script-location --user -r requirements.txt \
    && python3 manage.py collectstatic --noinput \
