"""
master_pipeline.py – run the entire pipeline end-to-end:
  1. Data ingestion + augmentation
  2. Feature engineering
  3. Model training (RF + XGBoost + LightGBM)
  4. Segmentation
  5. Retention simulation
"""
import sys, os, logging
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def main():
    log.info("=" * 60)
    log.info("STEP 1/5 – Data Ingestion & Augmentation")
    log.info("=" * 60)
    from src.data_ingestion.ingest import run as ingest_run
    ingest_run()

    log.info("=" * 60)
    log.info("STEP 2/5 – Feature Engineering")
    log.info("=" * 60)
    from src.features.build_features import run as feat_run
    feat_run()

    log.info("=" * 60)
    log.info("STEP 3/5 – Model Training")
    log.info("=" * 60)
    from src.models.train_churn import run as train_run
    train_run()

    log.info("=" * 60)
    log.info("STEP 4/5 – Customer Segmentation")
    log.info("=" * 60)
    from src.models.segment import run as seg_run
    seg_run()

    log.info("=" * 60)
    log.info("STEP 5/5 – Retention Simulation")
    log.info("=" * 60)
    from src.models.retention_simulator import run as sim_run
    sim_run()

    log.info("\n✅  Pipeline complete. Run the dashboard with:")
    log.info("    venv\\Scripts\\streamlit.exe run src/dashboard/app.py")

if __name__ == "__main__":
    main()
