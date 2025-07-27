# PostgreSQL Setup and Restore
You can use docker-compose, docker with the following commands to set up a PostgreSQL container and restore the database from a backup file.

## Using docker-compose

First of all, create a `.env` file in the root directory of your project with the following content:

```env
POSTGRES_USER=your_username
POSTGRES_PASSWORD=your_password
POSTGRES_DB=your_database_name
PGPORT=5432
```

```bash
docker compose up -d
```