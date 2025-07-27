#!/usr/bin/env bash
set -e

if [ -f "$MINEROS_DB_BACKUP_FILEPATH" ]; then
    gunzip -c "$MINEROS_DB_BACKUP_FILEPATH" \
        | sed "/^CREATE ROLE /d;/ROLE /d;/^GRANT /d;/OWNER TO /d" \
        | psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"
fi

psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "
    CREATE TABLE mineros_init (
        done BOOLEAN NOT NULL DEFAULT TRUE
    );

    INSERT INTO mineros_init VALUES (DEFAULT);
"