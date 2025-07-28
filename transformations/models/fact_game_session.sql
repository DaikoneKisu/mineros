with standoff_game_sessions as (
    select
        wallet_movement."id" as id,
        wallet_movement."amount" as amount,
        wallet."currency" as currency,
        wallet_movement."createdAt" as datetime,
        play."gameUuid" as game_id,
        wallet."userUuid" as player_id,
        '714720f5-27fe-4a03-be00-092835a35eed'::uuid as provider_id
    from vibra_game_movement
    inner join game_movement
        on vibra_game_movement."gameMovementId" = game_movement."walletMovementId"
    inner join wallet_movement
        on game_movement."walletMovementId" = wallet_movement.id
    inner join wallet
        on wallet_movement."walletId" = wallet.id
    inner join play
        on vibra_game_movement."playId" = play.id
    where
        wallet_movement."transactionType" = 'cut'
), domino_game_sessions as (
    select
        wallet_movement."id" as id,
        wallet_movement."amount" as amount,
        wallet."currency" as currency,
        wallet_movement."createdAt" as datetime,
        '57902ec7-fdb9-445f-b349-370cc9351708'::uuid as game_id,
        wallet."userUuid" as player_id,
        '692de0bd-20d9-4ad1-8d36-185858f35054'::uuid as provider_id
    from domino_game_movement
    inner join game_movement
        on domino_game_movement."gameMovementId" = game_movement."walletMovementId"
    inner join wallet_movement
        on game_movement."walletMovementId" = wallet_movement.id
    inner join wallet
        on wallet_movement."walletId" = wallet.id
    where
        wallet_movement."transactionType" = 'cut'
), animalito_game_sessions as (
    select
        wallet_movement."id" as id,
        wallet_movement."amount" as amount,
        wallet."currency" as currency,
        wallet_movement."createdAt" as datetime,
        'a0a6f409-2484-4560-9012-8023e5a24c9a'::uuid as game_id,
        wallet."userUuid" as player_id,
        '692de0bd-20d9-4ad1-8d36-185858f35054'::uuid as provider_id
    from animalito_game_movement
    inner join game_movement
        on animalito_game_movement."gameMovementId" = game_movement."walletMovementId"
    inner join wallet_movement
        on game_movement."walletMovementId" = wallet_movement.id
    inner join wallet
        on wallet_movement."walletId" = wallet.id
    where
        wallet_movement."transactionType" = 'cut'
), ludo_game_sessions as (
    select
        wallet_movement."id" as id,
        wallet_movement."amount" as amount,
        wallet."currency" as currency,
        wallet_movement."createdAt" as datetime,
        'a0a6f409-2484-4560-9012-8023e5a24c9a'::uuid as game_id,
        wallet."userUuid" as player_id,
        '692de0bd-20d9-4ad1-8d36-185858f35054'::uuid as provider_id
    from ludo_game_movements
    inner join game_movement
        on ludo_game_movements."gameMovementId" = game_movement."walletMovementId"
    inner join wallet_movement
        on game_movement."walletMovementId" = wallet_movement.id
    inner join wallet
        on wallet_movement."walletId" = wallet.id
    where
        wallet_movement."transactionType" = 'cut'
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