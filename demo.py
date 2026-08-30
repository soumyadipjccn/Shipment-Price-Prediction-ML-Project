from shipment.pipline.training_pipeline import TrainPipeline
from shipment.logger import logging

if __name__ == "__main__":
    try:
        logging.info("Starting training pipeline from demo.py...")
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        print("Training pipeline executed successfully!")
    except Exception as e:
        logging.error(f"Training pipeline failed: {e}")
        print(f"Error executing training pipeline: {e}")
