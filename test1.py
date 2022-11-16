from main import parse_args
from util.default_args import set_model_defaults
from datasets import build_dataset
from torch.utils.data import DataLoader
import util.misc as utils
from compute_individual_losses import compute_individual_losses, build_model_with_individual_criterion


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

model, individual_criterion, postprocessors = build_model_with_individual_criterion(args)
model.to("cuda:0")

samples, targets = next(iter(val_loader))
samples, targets = samples.to("cuda:0"), [{k: v.to("cuda:0") for k, v in t.items()} for t in targets]
outputs = model(samples)
print(outputs["dec_outputs"][-1].shape)

losses = compute_individual_losses(individual_criterion, outputs, targets)
print(losses.shape)

# model.zero_grad()
# losses.backward(inputs=outputs["dec_outputs"])
# print(outputs["dec_outputs"].grad[-1].shape)