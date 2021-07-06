import os
import disentanglement_lib.utils.hyperparams as h
import argparse
os.environ['WANDB_PROJECT'] = 'uncertainty'
os.environ['WANDB_TAGS'] = 'order'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('s', type=int, default=0)
parser.add_argument('e', type=int, default=0)
parser.add_argument('--skip', type=int, default=0)
args = parser.parse_args()

seed = h.sweep('train.random_seed', h.discrete(range(args.s, args.e)))

model = h.sweep('model.regularizers', h.discrete(['[@vae]', '[@beta_tc_vae]']))
model.append({
    'model.regularizers': '[@cascade_vae_c]',
    'cascade_vae_c.stage_steps': '5000',
})
dataset = [
    {'train.dataset': "\"'dsprites_full'\"", 'dataset.name': "\"'dsprites_full'\"",},
    # {'train.dataset': "\"'color_dsprites'\"", },
    # {'train.dataset': "\"'scream_dsprites'\"",},
    {'train.dataset': "\"'smallnorb'\"",'dataset.name': "\"'smallnorb'\"",},
    {'train.dataset': "\"'cars3d'\"",'dataset.name': "\"'cars3d'\""},
]
runs = h.product([seed, model, dataset])

general_config = {
    'train.eval_callbacks': '[@eval_mig]',
    'eval_mig.evaluation_steps': 1000,
    'train.training_steps': '50000',
}
metrics = " --metrics dci factor_vae_metric modularity_explicitness"
print(len(runs))
for i, run_args in enumerate(runs):
    if i < args.skip: continue
    run_args.update(general_config)
    args_str = " ".join([f"--{k}={v}" for k, v in run_args.items()])
    args_str += metrics
    print(args_str, f"{100 * i // len(runs)}%")
    # print(args_str)
    ret = os.system("dlib_run " + args_str)
    if ret != 0:
        exit(ret)
