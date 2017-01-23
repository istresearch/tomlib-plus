

def get_item_recur(key, obj):
    if key in obj:
        return obj[key]
    for k, v in obj.items():
        if isinstance(v,dict):
            item = get_item_recur(key, v)
            if item is not None:
                return item
