with standoff_games as (
    select
        game.uuid as id,
        substr(md5(random()::text), 0, 12) as name,
        lower(category.name) as type
    from game
    inner join game_category
        on game.uuid = game_category.game_uuid
    inner join category
        on game_category.category_id = category.id
    where
        category.status = 'active'
), elbetaso_games as (
    select
        *
    from (values
        ('57902ec7-fdb9-445f-b349-370cc9351708', 'domino', 'domino'),
        ('a0a6f409-2484-4560-9012-8023e5a24c9a', 'animalito', 'animalito'),
        ('16427564-7bf4-4948-afb9-77919729b93b', 'ludo', 'ludo')
    ) as games(id, name, type)
)
select 
    * 
from 
    standoff_games 
union
select
    *
from
    elbetaso_games