def create_parameter_list(ret, object_name, update_function):
    r = f"|||\n"
    r += f"|---|---|\n"
    for key, value in ret.parameter.items():
        r += f"|**{key}**|"
        if key in ret.parameter_choices:
            choices = ";".join([str(x) for x in ret.parameter_choices[key]])
            r += f"<|{{str({object_name}.parameter['{key}'])}}|selector|lov={choices}|dropdown|on_change={{{update_function}}}|>"
        else:
            r += f"<|{{str({object_name}.parameter['{key}'])}}|input|on_change={{{update_function}}}|>"
        r += "|\n"
    return r


def create_variable_parameter_list(ret, object_name, update_function):
    r = ""
    for i, x in enumerate(ret.all):
        r += f'<|part|render={{type({object_name}.current).__name__=="{type(x).__name__}"}}|\n'
        r += create_parameter_list(x, object_name + f".all[{i}]", update_function)
        r += f"|>\n"
    return r
