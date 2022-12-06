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


def df_to_highcharts_clustered_bar(df, colors=False, single_series = False):

    color_set = ['#2f7ed8', '#0d233a', '#8bbc21', '#910000', '#1aadce',
        '#492970', '#f28f43', '#77a1e5', '#c42525', '#a6c96a', '#B5CA92']

    import seaborn as sns
    palette = sns.color_palette("Set2", 20).as_hex()
    print(palette)


    df = df.round(2)

    columns = df.columns.tolist()
    rows = df.index.tolist()
    data = []

    for ixcol, col in enumerate(columns):
        values = []

        if single_series:
            for ixrow, row in enumerate(rows):
                values.append({
                    "y":df.iloc[ixrow, ixcol],
                    "color":palette[ixrow]
                })
   
        else:
            for ixrow, row in enumerate(rows):
                values.append(df.iloc[ixrow, ixcol])

        if colors:
            data.append({
                "name": df.columns[ixcol],
                "data": values,
                "stack": ixcol,
                "color":color_set[ixcol]
            })
        else:
            data.append({
                "name": df.columns[ixcol],
                "data": values,
                "stack": ixcol
            })
        
    json_data = {'rows':rows, 'columns':columns, 'data':data}


    response = json.dumps(json_data)

    return response


def df_to_highcharts_linechart(df, dualAxis=False):

    df = df.round(2)

    columns = df.columns.tolist()
    rows = df.index.tolist()
    data = []

    for ixcol, col in enumerate(columns):
        values = []

        for ixrow, row in enumerate(rows):
            values.append(df.iloc[ixrow, ixcol])

        if dualAxis and ixcol == 0:
            data.append({
                "name": df.columns[ixcol],
                "data": values,
                "yAxis":1
            })
        else:
            data.append({
                "name": df.columns[ixcol],
                "data": values,
            })
            
    json_data = {'rows':rows, 'columns':columns, 'data':data}

    response = json.dumps(json_data)

    return response