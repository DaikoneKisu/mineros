select
    users."uuid" as id,
    substr(md5(random()::text), 0, 12) as name,
    extract(year from age(CURRENT_DATE, users."birthDate")) as age
from public.user as users
where
    users."statusVerify" = 'verified' and
    users."isBlocked" = false
