def create_detailed_description(ret, object_name):
    r = ""
    for i, x in enumerate(ret.all):
        r += f'<|part|render={{type({object_name}.current).__name__=="{type(x).__name__}"}}|\n'
        r += x.__doc__
        r += f"|>\n"
    return r


def create_short_description(ret, object_name):
    r = ""
    for i, x in enumerate(ret.all):
        r += f'<|part|render={{type({object_name}.current).__name__=="{type(x).__name__}"}}|\n'
        r += x.__str__
        r += f"|>\n"
    return r
