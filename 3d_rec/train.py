import sys
import os
import time
import pandas as pd
import torch
import datasets
import models
from loggers import loggers
from util.util_print import str_error, str_stage, str_verbose, str_warning
from util import util_loadlib as loadlib
from util.util.io import overwrite as overwrite
import argparse


parser = argparse.ArgumentParser(description='Training code for 3D generation')
# Training settings
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




#Log and save directory
if opt.gpu:
    loadlib.set_gpu(opt.gpu)
    device = torch.device('cuda')
if opt.manual_seed is not None:
    loadlib.set_manual_seed(opt.manual_seed)

	
print(str_stage, "Set up log directory")
exprdir = '{}_{}_{}'.format(opt.net, opt.dataset, opt.lr)
exprdir += ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
logdir = os.path.join(opt.logdir, exprdir, str(opt.expr_id))


if opt.resume == 0:
    if os.path.isdir(logdir):
          print(
              str_warning, (
                    "Will remove experiment %d at\n\t%s\n"
                    "want to continue? (y/n)"
                ) % (opt.expr_id, logdir)
            )
          need_input = True
          while need_input:
              response = input().lower()
              if response in ('y', 'n'):
                  need_input = False
          if response == 'n':
              print(str_stage, "User decides to quit")
              sys.exit()
          os.system('rm -rf ' + logdir)

    os.system('mkdir -p ' + logdir)
else:
    assert os.path.isdir(logdir)
    opt_f_old = os.path.join(logdir, 'opt.pt')
    opt = overwrite(opt, opt_f_old)

	
# Save opt
torch.save(vars(opt), os.path.join(logdir, 'opt.pt'))
with open(os.path.join(logdir, 'opt.txt'), 'w') as logout:
    for k, v in vars(opt).items():
        logout.write('%20s\t%-20s\n' % (k, v))

opt.full_logdir = logdir
print(str_verbose, "Log directory set to: %s" % logdir)



print(str_stage, "Setting up loggers")
if opt.resume != 0 and os.path.isfile(os.path.join(logdir, 'best.pt')):
    try:
        prev_best_data = torch.load(os.path.join(logdir, 'best.pt'))
        prev_best = prev_best_data['loss_eval']
        del prev_best_data
    except KeyError:
        prev_best = None
else:
    prev_best = None
best_model_logger = loggers.ModelSaveLogger(
    os.path.join(logdir, 'best.pt'),
    period=1,
    save_optimizer=True,
    save_best=True,
    prev_best=prev_best
)
logger_list = [
    loggers.TerminateOnNaN(),
    loggers.ProgbarLogger(allow_unused_fields='all'),
    loggers.CsvLogger(
        os.path.join(logdir, 'epoch_loss.csv'),
        allow_unused_fields='all'
    ),
    loggers.ModelSaveLogger(
        os.path.join(logdir, 'nets', '{epoch:04d}.pt'),
        period=opt.save_net,
        save_optimizer=opt.save_net_opt
    ),
    loggers.ModelSaveLogger(
        os.path.join(logdir, 'checkpoint.pt'),
        period=1,
        save_optimizer=True
    ),
    best_model_logger,
]
logger = loggers.ComposeLogger(logger_list)


#Train
Model = models.get_model(opt.net)
model = Model(opt, logger)
model.to(device)
print("Total model parameters: {:,d}".format(model.num_parameters()))


init_epoch = 1
if opt.resume != 0:
    if opt.resume == -1:
        net_filename = os.path.join(logdir, 'checkpoint.pt')
    elif opt.resume == -2:
        net_filename = os.path.join(logdir, 'best.pt')
    else:
        net_filename = os.path.join(
            logdir, 'nets', '{epoch:04d}.pt').format(epoch=opt.resume)


    additional_values = model.load_state_dict(net_filename, load_optimizer='auto')
    try:
        init_epoch += additional_values['epoch']
    except KeyError as err:

        epoch_loss_csv = os.path.join(logdir, 'epoch_loss.csv')
        if opt.resume == -1:
            try:
                init_epoch += pd.read_csv(epoch_loss_csv)['epoch'].max()
            except pd.errors.ParserError:
                with open(epoch_loss_csv, 'r') as f:
                    lines = f.readlines()
                init_epoch += max([int(l.split(',')[0]) for l in lines[1:]])
        else:
            init_epoch += opt.resume

###################################################


start_time = time.time()
dataset = datasets.get_dataset(opt.dataset)
dataset_train = dataset(opt, mode='train', model=model)
dataset_vali = dataset(opt, mode='vali', model=model)
dataloader_train = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.workers,
    pin_memory=True,
    drop_last=True
)
dataloader_valid = torch.utils.data.DataLoader(
    dataset_vali,
    batch_size=opt.batch_size,
    num_workers=opt.workers,
    pin_memory=True,
    drop_last=True,
    shuffle=False
)


print(str_verbose, "Training batches per epoch: " + str(len(dataloader_train)))
print(str_verbose, "Testing batches: " + str(len(dataloader_valid)))

###################################################

if opt.epoch > 0:
    print(str_stage, "Training")
    model.train_epoch(
        dataloader_train,
        dataloader_eval=dataloader_valid,
        max_batches_per_train=opt.epoch_batches,
        epochs=opt.epoch,
        init_epoch=init_epoch,
        max_batches_per_eval=opt.eval_batches,
        eval_at_start=opt.eval_at_start
    )
