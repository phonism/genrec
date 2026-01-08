"""
Data utils
"""
from genrec.data.schemas import SeqBatch


def cycle(dataloader):
    """
    cycle dataloader never stop
    """
    while True:
        for data in dataloader:
            yield data