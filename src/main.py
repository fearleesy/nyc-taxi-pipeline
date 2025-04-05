from src.data_ingestion.streamer import batch_generator
from src.data_ingestion.sources import DataFetcher
from src.data_ingestion.storage import DatabaseManager
from src.utils.metadata import MetadataCalculator
from src.utils.logger import setup_logger
import yaml
import sys

def main(config_path="config/config.yaml"):
    logger = setup_logger("main")

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        source_type = config["data_input"]["source_type"]
        source_config = config["data_input"][source_type]
        
        try:
            if source_type == "csv":
                df = DataFetcher.from_csv(source_config["path"])
            elif source_type == "s3":
                df = DataFetcher.from_s3(
                    bucket=source_config["bucket"],
                    key=source_config["key"],
                    aws_access_key=source_config.get("aws_access_key"),
                    aws_secret_key=source_config.get("aws_secret_key"),
                    region=source_config.get("region")
                )
            elif source_type == "api":
                df = DataFetcher.from_api(
                    url=source_config["url"],
                    params=source_config.get("params"),
                    headers=source_config.get("headers"),
                    format=source_config.get("format", "json")
                )
            else:
                raise ValueError(f"Unknown source type: {source_type}")
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            return

        db_manager = DatabaseManager(
            db_path=config["storage"]["database"]
        )

        batch_iter = batch_generator(
            data=df,
            batch_size=config["data_input"]["batch_size"],
            delay=config["data_input"]["stream_delay"]
        )
        
        for batch in batch_iter:
            try:
                db_manager.save_batch(batch)
                logger.info(f"Saved batch with {len(batch)} records")
            except Exception as e:
                logger.error(f"Failed to save batch: {str(e)}")

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
