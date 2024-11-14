import argparse

from ultralytics import YOLO

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=None, required=True, type=str)
    parser.add_argument('--model', default=None, required=True, type=str)
    parser.add_argument('--batch-size', default=16, required=False, type=int)
    parser.add_argument('--device', default="0,1,2,3,4,5,6,7", required=False, type=str)
    parser.add_argument('--imgsz', default=1600, required=False, type=int)
    parser.add_argument('--split', default='val', required=False, type=str)
    args = parser.parse_args()

    # Load a pre-trained model
    model = YOLO(args.model)

    # Train the model
    validation_results = model.val(
        data=f'{args.data}.yaml',
        imgsz=args.imgsz,
        batch=args.batch_size,
        device=args.device,
        split=args.split,
    )
