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



def df_to_highcharts_heatmap(df):
    ''' Return a pandas dataframe as JSON object for highcharts heatmap '''
    
    columns = df.columns.tolist()
    rows = df.index.tolist()
    data = []
    
    for ixrow, row in enumerate(rows):
    
        for ixcol, col in enumerate(columns):
    
            data.append([ixcol, ixrow, df[col].iloc[ixrow]])
    
    json_data = {'rows':rows, 'columns':columns, 'data':data}

    response = json.dumps(json_data)

    return response


def df_to_highcharts_clustered_bar(df):

    df = df.round(2)
    
    columns = df.columns.tolist()
    rows = df.index.tolist()
    data = []
    print(rows)

    for ixcol, col in enumerate(columns):
        values = []

        for ixrow, row in enumerate(rows):
            values.append(df.iloc[ixrow, ixcol])

        data.append({
            "name": df.columns[ixcol],
            "data": values,
            "stack": ixcol
        })
    
    json_data = {'rows':rows, 'columns':columns, 'data':data}

    response = json.dumps(json_data)

    return response
