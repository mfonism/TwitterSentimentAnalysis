def clean_string(string):
    # our raw tweets were byte encoded before being saved to csv
    # so, they are wrapped in b''
    # strip this off of them
    return string.lstrip("b'").rstrip("'")
