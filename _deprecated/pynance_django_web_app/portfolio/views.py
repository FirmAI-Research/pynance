from django.shortcuts import render
from django.shortcuts import render
from matplotlib.font_manager import json_dump
import pandas as pd
import sys
import os
import json
import nasdaqdatalink
from pyparsing import line
from lib.calendar import Calendar
from db.postgres import Postgres
from lib import numeric
import yfinance

cal = Calendar()
from dateutil.relativedelta import relativedelta
import datetime
from tabulate import tabulate


# Create your views here.
def rebalance(request):
    return render(request, 'rebalance.html')



def optimize(request):
    return render(request, 'optimize.html')