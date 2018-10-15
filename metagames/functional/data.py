"""Working with experiment data."""
import numpy as np


def experiment_step_statistics(
    data, substep_keys=("grad_norm",), round_keys=("utility", "action_logit", "opponent_action_logit")
):
    """Aggregate data into per-play global-step statistics.

    Args:
        data: Data dictionary created by `Experiment.run`.
        substep_keys: Attributes to aggregate from the substep data.
        round_keys: Attributes to aggregate from the round data.

    Returns:
        A nested dictionary of player_name => attribute_key => statistic_type => values
    """
    keys = tuple(substep_keys) + tuple(round_keys)
    player_names = tuple(player.name for player in data["players"])
    if len(set(player_names)) != len(player_names):  # Non-unique names
        player_names = tuple("{:d}_{:s}".format(i, name) for i, name in enumerate(player_names))

    statistic_types = ("mean", "max", "min")
    statistics = {
        name: {key: {stat_type: [] for stat_type in statistic_types} for key in keys} for name in player_names
    }

    for step_data in data["steps"]:
        for player_name, player_step_data in zip(player_names, step_data["player_updates"]):
            values = {key: [] for key in keys}
            for player_substep_data in player_step_data:
                for key in substep_keys:
                    values[key].append(player_substep_data[key])

                for round_data in player_substep_data['rounds']:
                    for key in round_keys:
                        values[key].append(round_data[key])

            player_stats = statistics[player_name]
            for key, value_list in values.items():
                key_stats = player_stats[key]
                key_stats['mean'].append(np.mean(value_list))
                key_stats['min'].append(np.min(value_list))
                key_stats['max'].append(np.max(value_list))
    return statistics
