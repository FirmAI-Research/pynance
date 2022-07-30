import sys, os
import json
import pandas as pd 

# conf
cwd = os.getcwd()
parentdir = os.path.dirname(cwd)
fp = os.path.join(parentdir, 'secrets.json')
with open(fp, 'r') as f:
    data = json.load(f)



from sqlalchemy import create_engine

class Postgres:

    def __init__(self):
        self.user = data['rds_user']
        self.password = data['rds_password']
        self.host = 'pynance.ckdvk4iveujs.us-east-1.rds.amazonaws.com'
        self.port = 5432
        self.database = 'postgres' 
        self.engine = create_engine(url="postgresql://{0}:{1}@{2}:{3}/{4}".format(self.user, self.password, self.host, self.port, self.database ) )

    
    def get_connection(self):
        return create_engine(
            url="mysql+pymysql://{0}:{1}@{2}:{3}/{4}".format(
                self.user, self.password, self.host, self.port, self.database
            )
        )


    def append_without_duplicates(self, table_name, df):
        '''
        Append dataframe to table_name without duplicates.
        '''
        try:
            table_contents = pd.read_sql_table(table_name, self.engine) 
            df = df[~df.isin(table_contents)]
        except:
            pass
        
        df.to_sql(table_name, self.engine, if_exists='append', index=False)
