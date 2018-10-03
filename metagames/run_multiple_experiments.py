"""Script that lets you run experiments with different sets of
hyperparameters to record outcomes. Calls play.py with those hyperparams."""
from subprocess import call
import metagames.play as play


def run_compare_biases_settings():
    # for biases in ['0', '1']:
    #     run(biases=biases)
    # layer_sizes = ['10', '5']
    # run(layer_sizes=layer_sizes)
    # for game in ['OSPD', 'IPD']:
    #     run(game=game)
    # run(joint_optim='False')
    run(False,
        joint_optim=False,
        game='OSPD',
        layer_sizes=['10', '5'],
        DD_val='-2.9')


def run(command_instead_of_main, **kwargs):
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
    if command_instead_of_main:
        command = ['python3', 'play.py'] + arglist
        print('Running command', command)
        call(command)
    else:
        print('Running play.main with args:', argstring)
        play.main(arglist)




    # def run(# Set all hyperparams to default vals
    #
    #                 # Optimization
    #                 dont_diff_through_inner_opt = default_args.dont_diff_through_inner_opt,
    #                 weight_grad_paths = default_args.weight_grad_paths,
    #                 lr_out = default_args.lr_out,
    #                 lr_in = default_args.lr_in,
    #                 optim_algo = default_args.optim_algo,
    #                 joint_optim=default_args.joint_optim,
    #                 n_outer_opt = default_args.n_outer_opt,
    #                 n_inner_opt_range = default_args.n_inner_opt_range,
    #
    #                 # Game env
    #                 game = default_args.game,
    #                 net_type = default_args.net_type,
    #                 DD_val = default_args.DD_val,
    #                 CC_val = default_args.CC_val,
    #                 gamma = default_args.gamma,
    #
    #                 # Neural nets
    #                 layer_sizes = default_args.layer_sizes,
    #                 init_std = default_args.init_std,
    #                 seed = default_args.seed,
    #                 biases = default_args.biases,
    #                 biases_init = default_args.biases_init,
    #                 layers_wo_bias = default_args.layers_wo_bias,
    #
    #                 # Other
    #                 exp_group_name = default_args.exp_group_name,
    #                 plot_progres = default_args.plot_progress,
    #                 plot_every_n = default_args.plot_every_n
    #         ):
    #
    #     command = ['python', 'play.py',
    #
    #                # Optimization
    #                '--dont-diff-through-inner-opt', dont_diff_through_inner_opt,
    #                '--weight-grad-paths', weight_grad_paths,
    #                '--lr_out', lr_out,
    #                '--lr_in', lr_in,
    #                '--optim-algo', optim_algo,
    #                '--joint-optim', joint_optim,
    #                '--n-outer-opt', n_outer_opt,
    #                '--n-inner-opt-range', n_inner_opt_range,
    #
    #                # Game env
    #                '--game', game,
    #                '--net-type', net_type,
    #                '--DD-val', DD_val,
    #                '--CC-val', CC_val,
    #                '--gamma', gamma,
    #
    #                # Neural nets
    #                '--layer-sizes', layer_sizes,
    #                '--init-std', init_std,
    #                '--seed', seed,
    #                '--biases', biases,
    #                '--biases-init', biases_init,
    #                '--layers-wo-bias', layers_wo_bias,
    #
    #                # Other
    #                '--exp-group-name', exp_group_name,
    #                '--plot-progress', plot_progres,
    #                '--plot-every-n', plot_every_n
    #                ]
    #     print('Running command', ' '.join([str(c) for c in command]))
    #     call(command)
    #
    # return run

def run_example(run):
    run()

if __name__=="__main__":
    # default_args = play.parse_args()

    run_compare_biases_settings()