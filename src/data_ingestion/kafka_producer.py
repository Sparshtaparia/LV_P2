"""
Kafka Producer: Simulates streaming real-time event logs to Kafka.
"""
import sys, os, time, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import logging
from confluent_kafka import Producer
from src.data_ingestion.ingest import generate_synthetic

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

KAFKA_BROKER = os.getenv("KAFKA_BROKER_URL", "localhost:9092")
TOPIC = "customer_events"

def delivery_report(err, msg):
    if err is not None:
        log.error(f"Message delivery failed: {err}")
    else:
        log.debug(f"Message delivered to {msg.topic()} [{msg.partition()}]")

def stream_events():
    conf = {'bootstrap.servers': KAFKA_BROKER}
    
    try:
        producer = Producer(conf)
    except Exception as e:
        log.error(f"Failed to connect to Kafka at {KAFKA_BROKER}: {e}")
        return

    log.info(f"Starting Kafka producer to {KAFKA_BROKER} on topic '{TOPIC}'...")

    while True:
        try:
            # Generate 1 synthetic record per tick
            df = generate_synthetic(1)
            record = df.iloc[0].to_dict()
            
            # Serialize
            data_str = json.dumps(record)
            
            # Produce
            producer.produce(TOPIC, key=str(record.get('customerID', 'unknown')), value=data_str, callback=delivery_report)
            producer.poll(0)
            
            log.info(f"Produced event for customer {record.get('customerID')}")
            time.sleep(2)  # Emit every 2 seconds
            
        except KeyboardInterrupt:
            log.info("Stopping producer...")
            break
        except Exception as e:
            log.error(f"Error producing event: {e}")
            time.sleep(5)

    producer.flush()

if __name__ == "__main__":
    stream_events()
