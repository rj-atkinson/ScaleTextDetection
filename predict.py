import argparse
from ultralytics import YOLO

# Argument parsing
parser = argparse.ArgumentParser(description="Run YOLO model prediction and print results.")
parser.add_argument('--model', type=str, required=True, help="Path to the YOLO PyTorch model file")
parser.add_argument('--source', type=str, required=True, help="Path to the image or video file")
args = parser.parse_args()

if __name__ == '__main__':
    # Load the model from the given file path
    model = YOLO(args.model)

    # Run prediction on the given source (image or video)
    results = model.predict(source=args.source)

    # Print detailed results for each prediction
    for result in results:
        print(f"Detected {len(result.boxes)} objects.")
        for box in result.boxes:
            print(f"Class: {box.cls}, Confidence: {box.conf}, Bounding Box: {box.xyxy}")
