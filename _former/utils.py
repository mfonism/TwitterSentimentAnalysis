import re


pat_1 = r"(?:\@|https?\://)\S+"
pat_2 = r"#\w+ ?"
combined_pat = r"|".join((pat_1, pat_2))

www_pat = r"www.[^ ]+"

html_tag_pat = r"<[^>]+>"

negations_ = {
    "isn't": "is not",
    "can't": "can not",
    "couldn't": "could not",
    "hasn't": "has not",
    "hadn't": "had not",
    "won't": "will not",
    "wouldn't": "would not",
    "aren't": "are not",
    "haven't": "have not",
    "doesn't": "does not",
    "didn't": "did not",
    "don't": "do not",
    "shouldn't": "should not",
    "wasn't": "was not",
    "weren't": "were not",
    "mightn't": "might not",
    "mustn't": "must not",
}
negations_pat = re.compile(r"\b(" + "|".join(negations_.keys()) + r")\b")
