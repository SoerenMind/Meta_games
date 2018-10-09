"""Script that lets you run experiments with different sets of
hyperparameters to record outcomes. Calls play.py with those hyperparams."""
import logging
import os

import metagames.play as play
import numpy as np


def run_joint_vs_alt_and_lr():
    independent_vars = ['joint_optim', 'lr_out', 'seed']    # For graph titles

    # Independent vars
    seed_opt = ['0', '1', '2', '3', '4']
    joint_optim_opt = ['True', 'False']
    lr_out_opt = ['1.', '0.1', '0.01']

    # Fixed hyperparams
    n_outer_opt = '150000'
    lr_in = '10'

    # File handling
    exp_group_name = '2018-10-08 18h: JOINT, LR'
    foldername = exp_group_name
    start_logger(exp_group_name)

    num_runs = np.product([len(opt) for opt in [seed_opt, lr_out_opt, joint_optim_opt]])
    run_counter = 0

    for lr_out in lr_out_opt:
        for joint_optim in joint_optim_opt:
            for seed in seed_opt:
                run_counter += 1
                print('Run %i / %i' %(run_counter, num_runs))
                run(independent_vars=independent_vars,
                    seed=seed,
                    lr_out=lr_out,
                    joint_optim=joint_optim,

                    exp_group_name=exp_group_name,
                    n_outer_opt=n_outer_opt,
                    foldername=foldername,
                    lr_in=lr_in,
                    )


def run(**kwargs):
    """Runs play.py with hyperparams given by kwargs."""
    arglist = []
    for name, value in sorted(kwargs.items()):
        # Handle flags set to True or False
        if value in ['False', False]:
            continue
        arglist.append('--{name}'.format(name=name.replace('_', '-')))
        if value in ['True', True]:
            continue

        # Append value unless value is (non-string) iterable, then append contents
        try:
            assert not isinstance(value, str)
            for v in value:
                arglist.append('{v}'.format(v=str(v)))
        except (AssertionError, TypeError):
            arglist.append('{value}'.format(value=str(value)))

    argstring = ' '.join(arglist)
    try:
        print('Running play.main with args:', argstring)
        play.main(arglist)
    except Exception as e:
        print('Experiment failed')
        logging.exception('Experiment failed: ' + argstring)


def start_logger(exp_group_name):
    """For logging when experiment runs fail"""
    data_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    pathname = os.path.join(data_directory, 'logs')
    os.makedirs(pathname, exist_ok=True)
    logging.basicConfig(filename=os.path.join(pathname, exp_group_name) + '.log', level=logging.DEBUG)


if __name__=="__main__":
    # run_compare_biases_settings()
    run_joint_vs_alt_and_lr()
