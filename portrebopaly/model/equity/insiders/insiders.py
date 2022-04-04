import numpy as np
import pandas as pd

from vendors.nasdaq import Nasdaq
from vendors.nasdaq import Insiders as CoreUsInsiders


class Insiders(Nasdaq):

    def __init__(self):
        super().__init__()

        self.df = CoreUsInsiders().get()

