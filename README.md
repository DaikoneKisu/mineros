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
docker-compose up -d
```

## Using Docker

```pwsh
docker volume create mineros-volume
docker run -d -p 5432:5432 --name mineros -v mineros-volume:/var/lib/postgresql/data -e POSTGRES_PASSWORD=postgres postgres:15
docker cp <backup_file> mineros:/home
docker exec -i -t mineros bash
```

Once you are inside the container, create the database and restore it using:

```bash
psql -U postgres -c "CREATE DATABASE mineros;"
pg_restore -U postgres -d mineros /home/<backup_file>
```
