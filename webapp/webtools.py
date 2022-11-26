import json
import pandas as pd

def df_to_dt(df, fp):
    ''' Return pandas dataframe as JSON object for Jquery Datatables '''
    
    with open (fp, 'w') as f:
     
        response = df.to_dict(orient='split') 
     
        response['columns'] = [{"title":x} for x in response['columns'] ]
     
        response = json.dumps(response) # use dumps to ensure double quotes
     
        f.write(response)
    
    return response



def to_highcharts():
    pass