FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

WORKDIR /app

RUN pip install --no-cache-dir poetry

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false

RUN poetry install --no-interaction --no-ansi --no-root --with dev

COPY . .

RUN poetry install --no-interaction --no-ansi --only-root

EXPOSE 8000

CMD ["marker_server", "--host", "0.0.0.0", "--port", "8000"]