import logging

logging.basicConfig(level=logging.INFO)

from ml.enrichment_engine import train_enrichment_model

def main():
    print("Starting Enrichment Engine training test...")
    try:
        metrics = train_enrichment_model()
        print("\n--- Training Completed ---")
        print(f"Metrics: {metrics}")
    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == '__main__':
    main()
