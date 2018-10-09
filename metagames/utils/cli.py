"""Command line / script utilities."""
import argparse


def _map_default(default, options_dict, nargs):
    try:
        if isinstance(nargs, str) and nargs in '*+':
            return [options_dict[value] for value in default]
        else:
            return options_dict[default]
    except KeyError:
        if default is None:
            return None
        raise


class DictLookupAction(argparse.Action):
    """Argparse action that allows only keys from a given dictionary.

    The dictionary should be passed as the argument to `choices`.
    The argument to `default` is used as a key in the dictionary.
    """

    def __init__(
        self,
        option_strings,
        dest,
        nargs=None,
        default=None,
        choices=None,
        required=False,
        help=None,  # pylint: disable=redefined-builtin
        metavar=None,
    ):
        if choices is None:
            raise ValueError("Must set choices to the lookup dict.")
        self.dict = choices
        default_value = _map_default(default, self.dict, nargs)

        super().__init__(
            option_strings,
            dest,
            nargs=nargs,
            default=default_value,
            choices=self.dict.keys(),
            required=required,
            help=help,
            metavar=metavar,
        )

    def __call__(self, parser, namespace, values, option_string=None):
        if self.nargs in (None, "?"):
            mapped_values = self.dict[values]
        else:
            mapped_values = [self.dict[v] for v in values]
        setattr(namespace, self.dest, mapped_values)
