FROM ghcr.io/astral-sh/uv:debian

RUN apt-get update && apt-get install -y build-essential libpq-dev

WORKDIR /mineros
COPY . .

RUN uv sync --locked

ENTRYPOINT ["uv", "run", "mineros"]