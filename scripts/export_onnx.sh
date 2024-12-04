pt_model_path=/workspace/models/hantian/yolo-doclaynet/yolov10b-doclaynet.pt
yolo export model=${pt_model_path} format=onnx dynamic=True opset=12
