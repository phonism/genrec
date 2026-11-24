"""CLI alias for the shared GenRec CTR trainer."""
from genrec.modules.utils import parse_config
from genrec.trainers.ranking_trainer import train

if __name__ == "__main__":
    parse_config()
    train()
