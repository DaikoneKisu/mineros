FROM postgres:alpine

WORKDIR /mineros/db
COPY ./db/betaso-backup.sql.gz .

WORKDIR /docker-entrypoint-initdb.d
COPY ./db/betaso-init.sh .

ENV BETASO_DB_BACKUP_FILEPATH="/mineros/db/betaso-backup.sql.gz"