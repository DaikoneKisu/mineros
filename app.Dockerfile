FROM ghcr.io/astral-sh/uv:alpine

WORKDIR mineros
COPY . .

RUN uv sync --locked

ENTRYPOINT ["uv", "run", "main.py"]