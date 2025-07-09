import yaml
import time
import logging
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError, NoBrokersAvailable

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger("KafkaTopicCreator")

CONFIG_PATH = "config/config.yaml"

def create_topics_from_config():
    """Reads the config file and creates Kafka topics for each symbol and timeframe."""
    try:
        with open(CONFIG_PATH, 'r') as f: config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"FATAL: Configuration file not found at {CONFIG_PATH}."); return

    kafka_cfg = config.get('kafka', {})
    producer_cfg = config.get('producer', {})
    processing_cfg = config.get('data_processing', {})
    
    bootstrap_servers = kafka_cfg.get('bootstrap_servers')
    topic_prefix = kafka_cfg.get('topic_market_data_prefix', 'market.data.raw.')
    symbols = producer_cfg.get('symbols_to_fetch', [])
    timeframes = processing_cfg.get('feature_engineer_timeframes', [])

    if not all([bootstrap_servers, symbols, timeframes]):
        logger.error("Config missing bootstrap_servers, symbols_to_fetch, or timeframes."); return

    admin_client = None
    for i in range(5):
        try:
            admin_client = KafkaAdminClient(bootstrap_servers=bootstrap_servers, client_id='topic_creator_admin')
            logger.info(f"Successfully connected to Kafka Admin at {bootstrap_servers}"); break
        except NoBrokersAvailable:
            logger.warning(f"Kafka not ready. Retrying in 5 seconds... ({i+1}/5)"); time.sleep(5)
    
    if not admin_client:
        logger.error("Failed to connect to Kafka. Please ensure it's running via `docker-compose up`."); return

    topics_list = [NewTopic(name=f"{topic_prefix}{tf}-{s}", num_partitions=1, replication_factor=1)
                   for s in symbols for tf in timeframes]

    logger.info(f"Attempting to create {len(topics_list)} topics...")
    try:
        admin_client.create_topics(new_topics=topics_list, validate_only=False)
        logger.info("Topic creation requests sent.")
    except TopicAlreadyExistsError as e:
        logger.warning(f"Some topics already existed, which is okay: {e}")
    except Exception as e:
        logger.error(f"Failed to create topics: {e}")
    finally:
        admin_client.close()
        logger.info("Topic creation process finished.")

if __name__ == "__main__":
    create_topics_from_config()