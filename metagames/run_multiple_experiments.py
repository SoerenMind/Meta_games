"""Script that lets you run experiments with different sets of
hyperparameters to record outcomes. Calls play.py with those hyperparams."""
import metagames.play as play
import numpy as np


def run_compare_biases_settings():
    independent_vars = ['seed', 'biases', 'biases_init', 'layers_wo_bias']

    seed_opt = ['0', '1', '2']
    # biases_opt = ['0', '1']
    biases_init_opt = ['-1', '0', '1', 'normal']
    layers_wo_bias_opt = [[], ['0'], ['0', '1'], ['1']]

    exp_group_name = '2018-10-04 19h: Biases options'
    foldername = exp_group_name
    n_outer_opt = '15000'
    lr_out = '0.05'


    num_runs = np.product([len(opt) for opt in [seed_opt, biases_init_opt, layers_wo_bias_opt]])
    run_counter = 0

    for layers_wo_bias in layers_wo_bias_opt:
        # for biases in biases_opt:
        for biases_init in biases_init_opt:
            for seed in seed_opt:
                run_counter += 1
                print('Run %i / %i' %(run_counter, num_runs))
                run(independent_vars=independent_vars,
                    seed=seed,
                    layers_wo_bias=layers_wo_bias,
                    # biases=biases,
                    biases_init=biases_init,
                    exp_group_name=exp_group_name,
                    n_outer_opt=n_outer_opt,
                    lr_out=lr_out,
                    foldername=foldername)


def run_compare_learning_rates_and_payoffs():
    # TODO(sorenmind): log time in filenames and log and errors somewhere.
    independent_vars = ['lr_out', 'lr_in', 'CC_val', 'DD_val', 'seed']

    seed_opt = ['2', '3', '4', '5']
    lr_out_opt = ['0.1', '0.01']
    lr_in_opt = ['1', '10']
    CCDD_val_opt = [('-0.1', '-2.9'),  ('-0.7', '-2.3'), ('-1', '-2'), ('-0.8', '-2.2')]

    exp_group_name = '2018-10-05 19h: 150k, lr_in, CC, DD'
    foldername = exp_group_name
    n_outer_opt = '150000'
    n_inner_opt_range = (0, 3)

    num_runs = np.product([len(opt) for opt in [seed_opt, lr_out_opt, lr_in_opt, CCDD_val_opt]])
    run_counter = 0

    for CC_val, DD_val in CCDD_val_opt:
        for lr_out in lr_out_opt:
            for lr_in in lr_in_opt:
                for seed in seed_opt:
                    run_counter += 1
                    print('Run %i / %i' %(run_counter, num_runs))
                    run(independent_vars=independent_vars,
                        seed=seed,
                        lr_out=lr_out,
                        lr_in=lr_in,
                        CC_val=CC_val,
                        DD_val=DD_val,
                        exp_group_name=exp_group_name,
                        n_outer_opt=n_outer_opt,
                        foldername=foldername,
                        n_inner_opt_range=n_inner_opt_range)


def run(**kwargs):
    """Runs play.py with hyperparams given by kwargs.
    command_instead_of_main sets whether the play.main() function is called or play is run by
    a new python instance (so if one run crashes the present script continues with the other runs).
    """
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
    except:
        print('Experiment failed')


if __name__=="__main__":
    # run_compare_biases_settings()
    run_compare_learning_rates_and_payoffs()
