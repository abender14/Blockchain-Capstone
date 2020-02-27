# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 23:15:40 2020

@author: alexb
"""

import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'RevolvingUtilizationOfUnsecuredLines':1.1,
                            'age':30,
                            'NumberOfTime30-59DaysPastDueNotWorse':1,
                            'DebtRatio':1.5,
                            'MonthlyIncome':3000.0,
                            'NumberOfOpenCreditLinesAndLoans':1,
                            'NumberOfTimes90DaysLate':1,
                            'NumberRealEstateLoansOrLines':1,
                            'NumberOfTime60-89DaysPastDueNotWorse':1,
                            'NumberOfDependents':1})

print(r.json())
