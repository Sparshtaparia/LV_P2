"""
Kafka Consumer: Consumes real-time event logs from Kafka and writes to the DB.
"""
import sys, os, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import logging
import pandas as pd
from confluent_kafka import Consumer, KafkaError
from src.data_ingestion.ingest import save_to_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

KAFKA_BROKER = os.getenv("KAFKA_BROKER_URL", "localhost:9092")
TOPIC = "customer_events"
GROUP_ID = "churn_pipeline_group"

def consume_events():
    conf = {
        'bootstrap.servers': KAFKA_BROKER,
        'group.id': GROUP_ID,
        'auto.offset.reset': 'earliest'
    }

    try:
        consumer = Consumer(conf)
    except Exception as e:
        log.error(f"Failed to connect to Kafka at {KAFKA_BROKER}: {e}")
        return

    consumer.subscribe([TOPIC])
    log.info(f"Starting Kafka consumer on '{TOPIC}'...")

    batch = []
    batch_size = 10  # Write to DB every 10 records for demonstration

    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    log.error(msg.error())
                    break

            try:
                data = json.loads(msg.value().decode('utf-8'))
                batch.append(data)
                
                if len(batch) >= batch_size:
                    df = pd.DataFrame(batch)
                    save_to_db(df)
                    log.info(f"Saved batch of {len(batch)} records to DB.")
                    batch = []
                    
            except Exception as e:
                log.error(f"Error processing message: {e}")

    except KeyboardInterrupt:
        log.info("Stopping consumer...")
    finally:
        consumer.close()

if __name__ == "__main__":
    consume_events()
