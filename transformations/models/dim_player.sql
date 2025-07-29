select
    users."uuid" as id,
    substr(md5(random()::text), 0, 12) as name,
    extract(year from age(CURRENT_DATE, users.birth_date)) as age
from public.user as users
where
    users.status_verify = 'verified' and
    users.is_blocked = false
