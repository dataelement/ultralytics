import argparse
import os
from pathlib import Path

# import debugpy

# debugpy.listen(("localhost", 9501))
# print("Waiting for debugger attach")
# debugpy.wait_for_client()
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG_PATH

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=None, required=True, type=str)
    parser.add_argument('--model_path', default=None, required=True, type=str)
    parser.add_argument('--cfg', default=DEFAULT_CFG_PATH, required=False, type=str)
    parser.add_argument('--reg_max', default=16, required=False, type=int)
    parser.add_argument('--epoch', default=None, required=True, type=int)
    parser.add_argument('--optimizer', default='auto', required=False, type=str)
    parser.add_argument('--momentum', default=0.937, required=False, type=float)
    parser.add_argument('--lr0', default=0.02, required=False, type=float)
    parser.add_argument('--warmup-epochs', default=3.0, required=False, type=float)
    parser.add_argument('--batch-size', default=16, required=False, type=int)
    parser.add_argument('--image-size', default=None, required=True, type=int)
    parser.add_argument('--mosaic', default=1.0, required=False, type=float)
    parser.add_argument('--close_mosaic', default=10, required=False, type=int)
    parser.add_argument('--pretrain', default=None, required=False, type=str)
    parser.add_argument('--val', default=1, required=False, type=int)
    parser.add_argument('--plot', default=0, required=False, type=int)
    parser.add_argument('--project', default=None, required=True, type=str)
    parser.add_argument('--resume', action=argparse.BooleanOptionalAction)
    parser.add_argument('--workers', default=8, required=False, type=int)
    parser.add_argument('--device', default="0", required=False, type=str)
    parser.add_argument('--save-period', default=10, required=False, type=int)
    parser.add_argument('--patience', default=100, required=False, type=int)
    args = parser.parse_args()

    # Load a pre-trained model
    model = YOLO(args.model_path)

    # whether to val during training
    if args.val:
        val = True
    else:
        val = False

    # whether to plot
    if args.plot:
        plot = True
    else:
        plot = False

    # Train the model
    name = f"{Path(args.model_path).stem}_{args.data}_epoch{args.epoch}_imgsz{args.image_size}_bs{args.batch_size}_reg_max{args.reg_max}"
    results = model.train(
        data=f'{args.data}.yaml',
        cfg=args.cfg,
        epochs=args.epoch,
        warmup_epochs=args.warmup_epochs,
        lr0=args.lr0,
        optimizer=args.optimizer,
        momentum=args.momentum,
        imgsz=args.image_size,
        mosaic=args.mosaic,
        batch=args.batch_size,
        device=args.device,
        workers=args.workers,
        plots=plot,
        reg_max=args.reg_max,
        exist_ok=False,
        val=val,
        resume=args.resume,
        save_period=args.save_period,
        patience=args.patience,
        project=args.project,
        name=name,
        close_mosaic=args.close_mosaic,
    )
