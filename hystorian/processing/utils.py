def dict_to_list(d, join_str="/", parent_key=""):
    if isinstance(d, dict):
        items = []
        for k, v in d.items():
            new_key = join_str.join(filter(None, [parent_key, k]))
            items.extend(dict_to_list(v, join_str, new_key))
        return items
    else:
        return [parent_key]
