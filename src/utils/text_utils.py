import re

from . import patterns


def clean_tweet(text):
    # remove URLs
    text = re.sub(patterns.combined_pat, "", text)
    text = re.sub(patterns.www_pat, "", text)
    # remove HTML tags
    text = re.sub(patterns.html_tag_pat, "", text)
    # change to lowercase and
    # replace contracted negations with longer form
    text = text.lower()
    text = patterns.negations_pat.sub(lambda x: patterns.negations_[x.group()], text)
    # keep only the letters
    text = re.sub("[^a-zA-Z]", " ", text)
    # remove randos
    text = re.sub(patterns.rando_pat, "", text)
    # tokenize
    return text


def strip_byte(string):
    # our raw tweets were byte encoded before being saved to csv
    # so, they are wrapped in b''
    # strip this off of them
    return string.lstrip("b'").rstrip("'")
