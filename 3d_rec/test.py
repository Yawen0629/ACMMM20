import os
import time
from shutil import rmtree
from tqdm import tqdm
import torch
import datasets
import models
from util.util_print import str_error, str_stage, str_verbose
import util.util_loadlib as loadlib
from loggers import loggers
import argparse


parser = argparse.ArgumentParser(description='Test code for 3D generation')
# Dataset IO
parser.add_argument('--input_rgb', type=str, required=True,
					help="Input RGB filename")
parser.add_argument('--input_mask', type=str, required=True,
					help=("Corresponding mask filename, it can also be obtained by setting a threshold"))

# Network
parser.add_argument('--net_file', type=str, required=True,
					help="Path to the trained network")

# Output
parser.add_argument('--output_dir', type=str, required=True,
					help="Output directory")
parser.add_argument('--overwrite', action='store_true',
					help="Whether to overwrite the output folder if it exists")

###################################################
parser.add_argument('--gpu', default='0', type=str,
					help='which gpu')
parser.add_argument('--manual_seed', type=int, default=None,
					help='manual seed')
parser.add_argument('--resume', type=int, default=0,
					help='resume training or not. 0 for scratch, -1 for last and -2 for prev best.')
parser.add_argument(
	'--suffix', default='', type=str,
	help="Suffix for `logdir` that will be formatted with `opt`, e.g., '{classes}_lr{lr}'"
)
parser.add_argument('--epoch', type=int, default=0,
					help='epochs')

# Dataset load
parser.add_argument('--dataset', type=str, default=None,
					help='dataset')
parser.add_argument('--workers', type=int, default=4,
					help='number of loading workers')
parser.add_argument('--classes', default='plane', type=str,
					help='object class')
parser.add_argument('--batch_size', type=int, default=16,
					help='training batch size')
parser.add_argument('--epoch_batches', default=None, type=int, help='number of batches used per epoch')
parser.add_argument('--eval_batches', default=None,
					type=int, help='max number of batches used for evaluation per epoch')
parser.add_argument('--eval_at_start', action='store_true',
					help='run evaluation before starting to train')
parser.add_argument('--log_time', action='store_true', help='adding time log')

# Network
parser.add_argument('--net', type=str, required=True,
					help='network type to use')

# Optimizer
parser.add_argument('--optim', type=str, default='adam',
					help='optimizer to use')
parser.add_argument('--lr', type=float, default=1e-4,
					help='learning rate')
parser.add_argument('--adam_beta1', type=float, default=0.5,
					help='beta1 of adam')
parser.add_argument('--adam_beta2', type=float, default=0.9,
					help='beta2 of adam')
parser.add_argument('--sgd_momentum', type=float, default=0.9,
					help="momentum factor of SGD")
parser.add_argument('--sgd_dampening', type=float, default=0,
					help="dampening for momentum of SGD")
parser.add_argument('--wdecay', type=float, default=0.0,
					help='weight decay')

# Logging and vis
parser.add_argument('--logdir', type=str, default=None,
					help='Root directory')
parser.add_argument('--log_batch', action='store_true',
					help='Log batch loss')
parser.add_argument('--expr_id', type=int, default=0,
					help='Experiment index. non-positive ones are overwritten by default. Use 0 for code test. ')
parser.add_argument('--save_net', type=int, default=50,
					help='Period of saving network weights')
parser.add_argument('--save_net_opt', action='store_true',
					help='Save optimizer state in regular network saving')
parser.add_argument('--vis_every_valid', default=100, type=int,
					help="Visualize every N epochs during validation")
parser.add_argument('--vis_every_train', default=100, type=int,
					help="Visualize every N epochs during training")
parser.add_argument('--vis_batches_valid', type=int, default=50,
					help="# batches to visualize during validation")
parser.add_argument('--vis_batches_train', type=int, default=50,
					help="# batches to visualize during training")
parser.add_argument('--vis_workers', default=4, type=int, help="# workers for the visualizer")
parser.add_argument('--vis_param_f', default=None, type=str,
					help="Parameter file read by the visualizer on every batch; defaults to 'visualize/config.json'")

opt = parser.parse_args()					
print(opt)

###################################################

print(str_stage, "Setting device")
if opt.gpu == '-1':
    device = torch.device('cpu')
else:
    loadlib.set_gpu(opt.gpu)
    device = torch.device('cuda')
if opt.manual_seed is not None:
    loadlib.set_manual_seed(opt.manual_seed)

###################################################

print(str_stage, "Setting up output dir")
output_dir = opt.output_dir
output_dir += ('_' + opt.suffix.format(**vars(opt))) \
    if opt.suffix != '' else ''
opt.output_dir = output_dir

if os.path.isdir(output_dir):
    if opt.overwrite:
        rmtree(output_dir)
    else:
        raise ValueError(str_error +
                         " %s already exists"
                         % output_dir)
os.makedirs(output_dir)

###################################################

print(str_stage, "Setting up loggers")
logger_list = [
    loggers.TerminateOnNaN(),
]
logger = loggers.ComposeLogger(logger_list)

###################################################

print(str_stage, "Setting up models")
Model = models.get_model(opt.net, test=True)
model = Model(opt, logger)
model.to(device)
model.eval()
print(model)
print("# model parameters: {:,d}".format(model.num_parameters()))

###################################################

print(str_stage, "Setting up data loaders")
start_time = time.time()
Dataset = datasets.get_dataset('test')
dataset = Dataset(opt, model=model)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batch_size,
    num_workers=opt.workers,
    pin_memory=True,
    drop_last=False,
    shuffle=False
)
n_batches = len(dataloader)
dataiter = iter(dataloader)


print(str_stage, "Now Testing!")
for i in tqdm(range(n_batches)):
    batch = next(dataiter)
    model.test_on_batch(i, batch)
