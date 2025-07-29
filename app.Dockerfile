FROM ghcr.io/astral-sh/uv:debian

RUN apt-get update && apt-get install -y build-essential libpq-dev

WORKDIR /mineros
COPY . .

RUN uv sync --locked

EXPOSE 8501

ENTRYPOINT ["uv", "run", "streamlit", "run", "src/mineros/app.py", "--server.port=8501", "--server.address=0.0.0.0"]