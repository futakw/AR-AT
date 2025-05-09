import ast


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
