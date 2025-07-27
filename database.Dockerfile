FROM postgres:alpine

WORKDIR /mineros/db
COPY ./db/backup.sql.gz .

WORKDIR /docker-entrypoint-initdb.d
COPY ./db/init.sh .

ENV MINEROS_DB_BACKUP_FILEPATH="/mineros/db/backup.sql.gz"