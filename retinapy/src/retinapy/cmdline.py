import argparse
import sys
import yaml

"""
Some useful functions to parse command line arguments.

The approach carried out here is inspired by the pytorch-image-models
project:
    https://github.com/rwightman/pytorch-image-models


Typically, you will do the following:

def create_parsers():
    yaml_parser = retinapy.cmdline.create_config_parser()
    cmdline_parser = argparse.ArgumentParser(description="My script.")
    cmdline_parser.add_argument("--foo", type=int, default=42)
    cmdline_parser.add_argument("--bar", type=str, default="hello")
    # etc...

opt, text = retinapy.cmdline.parse_args(cmdline_parser, yaml_parser)
"""


def create_yaml_parser():
    """Creates a parser with a single string `--config` argument.

    The created parser is intended for the second argument of the `parse_args`
    function.
    """
    config_parser = argparse.ArgumentParser(
        description="Config from YAML", add_help=False
    )
    config_parser.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        metavar="FILE",
        help="YAML config file to override argument defaults.",
    )
    return config_parser


def populate_from_yaml(parser, yaml_path):
    """Override parser defaults with values from a YAML file."""
    with open(yaml_path, "r") as f:
        args = yaml.safe_load(f)
    parser.set_defaults(**args)
    return parser


def parse_args(cmdline_parser, yaml_parser=None):
    """Parse command line arguments with optional YAML config file.

    Arguments are populated in the following order:
        1. Default values
        2. Config file
        3. Command line

    In addition to creating an opt object, this function also returns a yaml
    representation of the arguments as a string, which is useful for logging
    and creating a config file for future runs.
    """
    # The default behaviour of parse_args() is to parse sys.argv[1:]. Do
    # it explicitly here, so that we can optically use the yaml parser. It's
    # also nice anyway, as I always disliked the opacity of parse_args().
    remaining = sys.argv[1:]
    # First check if we have a config file to deal with.
    if yaml_parser:
        args, remaining = yaml_parser.parse_known_args()
        if args.config:
            populate_from_yaml(cmdline_parser, yaml_path=args.config)
    # Now the main parser.
    opt = cmdline_parser.parse_args(remaining)
    # Serialize the arguments.
    opt_text = yaml.safe_dump(opt.__dict__, default_flow_style=False)
    return opt, opt_text
