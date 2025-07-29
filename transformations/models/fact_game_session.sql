with standoff_game_sessions as (
    select
        wallet_movement.id as id,
        wallet_movement.amount as amount,
        wallet.currency as currency,
        wallet_movement.created_at as datetime,
        play.game_uuid as game_id,
        wallet.user_uuid as player_id,
        '714720f5-27fe-4a03-be00-092835a35eed' as provider_id
    from vibra_game_movement
    inner join game_movement
        on vibra_game_movement.game_movement_id = game_movement.wallet_movement_id
    inner join wallet_movement
        on game_movement.wallet_movement_id = wallet_movement.id
    inner join wallet
        on wallet_movement.wallet_id = wallet.id
    inner join play
        on vibra_game_movement.play_id = play.id
    where
        wallet_movement.transaction_type = 'cut'
), domino_game_sessions as (
    select
        wallet_movement.id as id,
        wallet_movement.amount as amount,
        wallet.currency as currency,
        wallet_movement.created_at as datetime,
        '57902ec7-fdb9-445f-b349-370cc9351708' as game_id,
        wallet.user_uuid as player_id,
        '692de0bd-20d9-4ad1-8d36-185858f35054' as provider_id
    from domino_game_movement
    inner join game_movement
        on domino_game_movement.game_movement_id = game_movement.wallet_movement_id
    inner join wallet_movement
        on game_movement.wallet_movement_id = wallet_movement.id
    inner join wallet
        on wallet_movement.wallet_id = wallet.id
    where
        wallet_movement.transaction_type = 'cut'
), animalito_game_sessions as (
    select
        wallet_movement.id as id,
        wallet_movement.amount as amount,
        wallet.currency as currency,
        wallet_movement.created_at as datetime,
        'a0a6f409-2484-4560-9012-8023e5a24c9a' as game_id,
        wallet.user_uuid as player_id,
        '692de0bd-20d9-4ad1-8d36-185858f35054' as provider_id
    from animalito_game_movement
    inner join game_movement
        on animalito_game_movement.game_movement_id = game_movement.wallet_movement_id
    inner join wallet_movement
        on game_movement.wallet_movement_id = wallet_movement.id
    inner join wallet
        on wallet_movement.wallet_id = wallet.id
    where
        wallet_movement.transaction_type = 'cut'
), ludo_game_sessions as (
    select
        wallet_movement.id as id,
        wallet_movement.amount as amount,
        wallet.currency as currency,
        wallet_movement.created_at as datetime,
        'a0a6f409-2484-4560-9012-8023e5a24c9a' as game_id,
        wallet.user_uuid as player_id,
        '692de0bd-20d9-4ad1-8d36-185858f35054' as provider_id
    from ludo_game_movements
    inner join game_movement
        on ludo_game_movements.game_movement_id = game_movement.wallet_movement_id
    inner join wallet_movement
        on game_movement.wallet_movement_id = wallet_movement.id
    inner join wallet
        on wallet_movement.wallet_id = wallet.id
    where
        wallet_movement.transaction_type = 'cut'
)
select
    *
from standoff_game_sessions
union
select
    *
from domino_game_sessions
union
select
    *
from animalito_game_sessions
union
select
    *
from ludo_game_sessions