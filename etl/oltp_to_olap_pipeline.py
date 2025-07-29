from datetime import datetime
import dlt
from dlt.sources.sql_database import sql_database

def oltp_to_olap_pipeline():
    db = sql_database(
        backend="connectorx",
        backend_kwargs={"return_type": "pandas"},
        reflection_level="full_with_precision",
    ).with_resources(
        "game",
        "game_category",
        "category",
        "user",
        "vibra_game_movement",
        "game_movement",
        "wallet_movement",
        "wallet",
        "play",
        "domino_game_movement",
        "animalito_game_movement",
        "ludo_game_movements"
    )
    db.max_table_nesting = 0

    initial_value = datetime.fromisoformat("1970-01-01T00:00:00Z")

    db.game.apply_hints(incremental=dlt.sources.incremental("updatedAt", initial_value=initial_value), primary_key="uuid")
    db.game_category.apply_hints(incremental=dlt.sources.incremental("updatedAt", initial_value=initial_value), primary_key="id")
    db.category.apply_hints(incremental=dlt.sources.incremental("updatedAt", initial_value=initial_value), primary_key="id")
    db.user.apply_hints(incremental=dlt.sources.incremental("updatedAt", initial_value=initial_value), primary_key="uuid")
    db.vibra_game_movement.apply_hints(primary_key="gameMovementId")
    db.game_movement.apply_hints(primary_key="walletMovementId")
    db.wallet_movement.apply_hints(incremental=dlt.sources.incremental("createdAt", initial_value=initial_value), primary_key="id")
    db.wallet.apply_hints(incremental=dlt.sources.incremental("createdAt", initial_value=initial_value), primary_key="id")
    db.play.apply_hints(incremental=dlt.sources.incremental("updatedAt", initial_value=initial_value), primary_key="id")
    db.domino_game_movement.apply_hints(primary_key="gameMovementId")
    db.animalito_game_movement.apply_hints(primary_key="gameMovementId")
    db.ludo_game_movements.apply_hints(primary_key="gameMovementId")

    raw_extract_pipeline = dlt.pipeline(
        pipeline_name="oltp_to_olap_raw_extract",
        destination="postgres",
        dataset_name="public",
        progress="alive_progress",
    )

    raw_load_info = raw_extract_pipeline.run(db, write_disposition="merge", loader_file_format="insert_values")
    print(raw_load_info)

    transformations_pipeline = dlt.pipeline(
        pipeline_name="oltp_to_olap_transformations",
        destination="postgres",
        dataset_name="olap",
        progress="alive_progress",
    )

    dbt = dlt.dbt.package(
        transformations_pipeline,
        "transformations",
    )

    models = dbt.run_all()

    for m in models:
        print(f"Model {m.model_name} has been materialized in {m.time} with status {m.status} and message {m.message}")

if __name__ == "__main__":
    oltp_to_olap_pipeline()
