#!/usr/bin/env bash
set -e

if [ -f "$BETASO_DB_BACKUP_FILEPATH" ]; then
    gunzip -c "$BETASO_DB_BACKUP_FILEPATH" \
        | sed "/^CREATE ROLE /d;/ROLE /d;/^GRANT /d;/OWNER TO /d" \
        | psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"
fi

psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "
    CREATE TABLE betaso_init (
        done BOOLEAN NOT NULL DEFAULT TRUE
    );

    INSERT INTO betaso_init VALUES (DEFAULT);
"