import os.path as osp
import sys

from sandbox.snn4hrl.warm_start import sync_s3
from rllab.misc.instrument import run_experiment_lite, stub
from rllab.misc.nb_utils import ExperimentDatabase
from sandbox.snn4hrl.warm_start.warm_starter import Warm_starter

mode = "ec2"  # not tried with ec2 yet..
local_root = "data/s3"
reload_prefix = "reload-"

# directory in S3 that contains all the experiments from where we want to re-load
log_dir = "trpo-Maze0-egoSwimmer-pre"

if mode == "local":
    n_parallel = 4
elif mode in ["ec2", "local_docker"]:
    n_parallel = 1
    sync_s3(log_dir)  # without specifying local_dir it will download to PROJECT_PATH +  "/data/s3", here just local
else:
    raise NotImplementedError

# all the exp are downloaded locally so I can do that
database = ExperimentDatabase(osp.join(local_root, log_dir), names_or_patterns='*')

exps = list(database.filter_experiments())

stub(globals())

for exp in exps:
    print('doing a warm start for: ', exp.params['exp_name'])
    analyzer = Warm_starter(
        log_dir=osp.join(log_dir, exp.params['exp_name']),
        # it will need again this to download in the ec2 instance the params
        local_root=local_root,
        batch_size=1e6,
        max_path_length=1e4,
    )

    exp_prefix = reload_prefix + '10e6Bs_1e4pl_' + log_dir
    exp_name = reload_prefix + exp.params['exp_name']

    for s in [10]:
        run_experiment_lite(
            stub_method_call=analyzer.run(),
            sync_s3_pkl=True,
            sync_s3_png=True,
            aws_config={'instance_type': 'm4.large', 'spot_price': '0.5'},
            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="last",
            seed=s,
            exp_prefix=exp_prefix,  # this will have the reload prefix to not overlap with the previous
            exp_name=exp_name,
            mode=mode,
            terminate_machine=True,
            pre_commands=[
                "which conda",
                "which python",
                "conda list -n rllab3",
                "conda install -f numpy -n rllab3 -y",
            ],
        )
        sys.exit()
