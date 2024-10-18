from ultralytics import YOLO

model = YOLO("./models/scale_text_detection_yolov8n_1729189598.pt")

results = model.predict(source='../v1/data-perfect/src/23-acaia-3/acaia-30025.jpg', save=True)

print(results)
