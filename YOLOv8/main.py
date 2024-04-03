from ultralytics import YOLO

datapath = '../Ludus_YOLO_Dataset'  # dataset location
model = YOLO('yolov8n')
result = model.train(data = datapath, epochs = 100, imgsz = 640)

