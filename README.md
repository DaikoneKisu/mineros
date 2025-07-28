# PostgreSQL Setup and Restore
You can use docker-compose, docker with the following commands to set up a PostgreSQL container and restore the database from a backup file.

## Using docker-compose

First of all, create a `.env` file in the root directory of your project with the following content:

```env
BETASO_POSTGRES_USER=
BETASO_POSTGRES_PASSWORD=
BETASO_POSTGRES_DB=
BETASO_PGPORT=

MINEROS_POSTGRES_USER=
MINEROS_POSTGRES_PASSWORD=
MINEROS_POSTGRES_DB=
MINEROS_PGPORT=

MINEROS_POSTGRES_URL=postgresql://${MINEROS_POSTGRES_USER}:${MINEROS_POSTGRES_PASSWORD}@localhost:${MINEROS_PGPORT}/${MINEROS_POSTGRES_DB}
```

```bash
docker compose up -d
```