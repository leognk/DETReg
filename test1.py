from main import parse_args
from util.default_args import set_model_defaults
from datasets import build_dataset
from torch.utils.data import DataLoader
import util.misc as utils
from compute_loss import *


args = f'''
--data_root /home/user/fiftyone
--dataset coco
--num_workers 1
--batch_size 2
--eval
'''

args = args.split()
args = parse_args(args)
set_model_defaults(args)

dataset_val = build_dataset(image_set='val', args=args)
val_loader = DataLoader(dataset_val, args.batch_size,
                                drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                pin_memory=True)

model, criterion, postprocessors = build_model(args)
model.to("cuda:0")

# Freeze all but the classifier.
for p in model.parameters():
    p.requires_grad = False
for p in model.class_embed[-1].parameters():
    p.requires_grad = True

samples, targets = next(iter(val_loader))
samples, targets = samples.to("cuda:0"), [{k: v.to("cuda:0") for k, v in t.items()} for t in targets]
outputs = model(samples)
print(outputs["dec_outputs"][-1].shape)

losses = compute_loss(criterion, outputs, targets)

model.zero_grad()
losses.backward()
print(outputs["all_pred_logits"].grad[-1].shape)