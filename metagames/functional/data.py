"""Working with experiment data."""
import numpy as np

SUBSTEP_KEYS = ("grad_norm",)
ROUND_KEYS = ("utility", "action_logit", "opponent_action_logit")
STATISTIC_FUNCTIONS = {"mean": np.mean, "min": np.min, "max": np.max}


def experiment_step_statistics(
    data, substep_keys=SUBSTEP_KEYS, round_keys=ROUND_KEYS, statistic_types=STATISTIC_FUNCTIONS.keys()
):
    """Aggregate data into per-play global-step statistics.

    Args:
        data: Data dictionary created by `Experiment.run`.
        substep_keys: Attributes to aggregate from the substep data.
        round_keys: Attributes to aggregate from the round data.
        statistic_types: A list of named statistics to calculate.
            Keys from `STATISTIC_FUNCTIONS`.


    Returns:
        A nested dictionary of player_name => attribute_key => statistic_type => values
    """
    keys = tuple(substep_keys) + tuple(round_keys)
    player_names = tuple(player.name for player in data["players"])
    if len(set(player_names)) != len(player_names):  # Non-unique names
        player_names = tuple("{:d}_{:s}".format(i, name) for i, name in enumerate(player_names))

    statistic_functions = {stat_type: STATISTIC_FUNCTIONS[stat_type] for stat_type in statistic_types}
    statistics = {
        name: {key: {stat_type: [] for stat_type in statistic_functions} for key in keys} for name in player_names
    }

    for step_data in data["steps"]:
        step_statistics = experiment_single_step_statistics(
            step_data, substep_keys=substep_keys, round_keys=round_keys, statistic_functions=statistic_functions
        )
        append_step_statistics(statistics, step_statistics, player_names)
    return statistics


def experiment_single_step_statistics(
    step_data,
    substep_keys=SUBSTEP_KEYS,
    round_keys=ROUND_KEYS,
    statistic_types=STATISTIC_FUNCTIONS.keys(),
    statistic_functions=None,
):
    """Aggregate statistics for one global step.

    Args:
        step_data: Data dictionary yielded by `Experiment.run_step`
        substep_keys: Attributes to aggregate from the substep data.
        round_keys: Attributes to aggregate from the round data.
        statistic_types: A list of named statistics to calculate.
            Keys from `STATISTIC_FUNCTIONS`.
        statistic_functions: A dictionary mapping named statistics to
            statistic functions. An alternative to `statistic_types`.

    Returns:
        A list of player_index => attribute_key => statistic_type => value
    """
    keys = tuple(substep_keys) + tuple(round_keys)
    if statistic_functions is None:
        statistic_functions = {stat_type: STATISTIC_FUNCTIONS[stat_type] for stat_type in statistic_types}

    player_statistics = []
    for player_step_data in step_data["player_updates"]:
        values = {key: [] for key in keys}
        for player_substep_data in player_step_data:
            for key in substep_keys:
                values[key].append(player_substep_data[key])

            for round_data in player_substep_data["rounds"]:
                for key in round_keys:
                    values[key].append(round_data[key])

        player_statistics.append(
            {
                key: {stat_type: stat_fn(value_list) for stat_type, stat_fn in statistic_functions.items()}
                for key, value_list in values.items()
            }
        )
    return player_statistics


def append_step_statistics(statistics, step_statistics, player_keys=None):
    """Append signle-step statistics to a dictionary of the same structure containing statistic lists."""
    if player_keys is None:
        keyed_step_statistics = enumerate(step_statistics)
    else:
        keyed_step_statistics = zip(player_keys, step_statistics)

    for player_name, player_step_statistics in keyed_step_statistics:
        try:
            player_stats = statistics[player_name]
        except KeyError:
            player_stats = {}
            statistics[player_name] = player_stats

        for key, key_step_statistics in player_step_statistics.items():
            try:
                key_stats = player_stats[key]
            except KeyError:
                key_stats = {}
                player_stats[key] = key_stats

            for stat_type, value in key_step_statistics.items():
                try:
                    key_stats[stat_type].append(value)
                except KeyError:
                    key_stats[stat_type] = [value]
