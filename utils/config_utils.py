import ast

import yaml


def parseStr(s):
    return ast.literal_eval(s) if s is not None else None


def str2intlist(s):
    s = s.replace("[", "").replace("]", "")
    values = s.split(",")
    l = [int(val) for val in values]
    return l


def str2floatlist(s):
    s = s.replace("[", "").replace("]", "")
    values = s.split(",")
    l = [float(val) for val in values]
    return l


def str2strlist(s):
    s = s.replace("[", "").replace("]", "")
    values = s.split(",")
    l = [str(val) for val in values]
    return l


def parse_with_config_file(parser):
    args = parser.parse_args()
    assert args.config_file.split(".")[-1] == "yaml", "only yaml file is allowed"

    config_file = args.config_file
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    input_args = []
    for key, value in config.items():
        if hasattr(args, key) is False:
            raise AttributeError("{} has no attribute '{}'".format(type(args), key))
        key = "--{}".format(key)
        input_args.extend([key, str(value)])

    args = parser.parse_args(input_args)
    args.config_file = config_file
    # setattr(args, 'exp_version', pathlib.Path(args.config_file).stem)
    # args.save_dir = os.path.join(args.save_dir, args.model_pre +
    #                              args.backbone.upper(), args.exp_version)
    #
    return args


def str2bool(var):
    return "t" in var.lower()


def split_str(string):
    splitted = string.split(",")
    fragments = []
    for each_splitted in splitted:
        use_list = each_splitted.split("-")
        fragments.append(use_list)
    return fragments


def split_comma(string):
    splitted = string.split(",")
    splitted = [spl.strip() for spl in splitted]

    return splitted
