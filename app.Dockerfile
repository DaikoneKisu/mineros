FROM ghcr.io/astral-sh/uv:alpine

RUN apk add build-base libpq libpq-dev

WORKDIR mineros
COPY . .

RUN uv sync --locked

ENTRYPOINT ["uv", "run", "mineros"]