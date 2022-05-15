
import re 
from dateutil.parser import parse  
import pandas as pd


def get_num_feats(data=None):
    '''
    Returns the numerical features in a data set
    Parameters:
    -----------
        data: DataFrame or named Series 
    Returns:
    -------
        List:
            A list of all the numerical features in a dataset.
    '''
    if data is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")

    num_features = data.select_dtypes(exclude=['object', 'datetime64']).columns

    return list(num_features)



def convert_dtype(df):
    '''
    Convert datatype of a feature to its original datatype.
    If the datatype of a feature is being represented as a string while the initial datatype is an integer or a float 
    or even a datetime dtype. The convert_dtype() function iterates over the feature(s) in a pandas dataframe and convert the features to their appropriate datatype
    
    Parameter:
    ---------------------------
    df: DataFrame, Series
        Dataset to convert data type
    
    Returns:
    -----------------
        DataFrame or Series.
    Example: 
    data = {'Name':['Tom', 'nick', 'jack'], 
            'Age':['20', '21', '19'],
            'Date of Birth': ['1999-11-17','20 Sept 1998','Wed Sep 19 14:55:02 2000']} 
     
    df = pd.DataFrame(data)
    df.info()
    >>> 
    <class 'pandas.core.frame.DataFrame'>
        RangeIndex: 3 entries, 0 to 2
        Data columns (total 3 columns):
        Name             3 non-null object
        Age              3 non-null object
        Date of Birth    3 non-null object
        dtypes: object(3)
        memory usage: 76.0+ bytes
    
    conv = convert_dtype(df)
    conv.info()
    >>> 
    <class 'pandas.core.frame.DataFrame'>
        RangeIndex: 3 entries, 0 to 2
        Data columns (total 3 columns):
        Name             3 non-null object
        Age              3 non-null int32
        Date of Birth    3 non-null datetime64[ns]
        dtypes: datetime64[ns](1), int32(1), object(1)
        memory usage: 88.0+ bytes
    '''
    if df.isnull().any().any() == True:
        raise ValueError("DataFrame contain missing values")
    else:
        i = 0
        changed_dtype = []
        #Function to handle datetime dtype
        def is_date(string, fuzzy=False):
            try:
                parse(string, fuzzy=fuzzy)
                return True
            except ValueError:
                return False
            
        while i <= (df.shape[1])-1:
            val = df.iloc[:,i]
            if str(val.dtypes) =='object':
                val = val.apply(lambda x: re.sub(r"^\s+|\s+$", "",x, flags=re.UNICODE)) #Remove spaces between strings
        
            try:
                if str(val.dtypes) =='object':
                    if val.min().isdigit() == True: #Check if the string is an integer dtype
                        int_v = val.astype(int)
                        changed_dtype.append(int_v)
                    elif val.min().replace('.', '', 1).isdigit() == True: #Check if the string is a float type
                        float_v = val.astype(float)
                        changed_dtype.append(float_v)
                    elif is_date(val.min(),fuzzy=False) == True: #Check if the string is a datetime dtype
                        dtime = pd.to_datetime(val)
                        changed_dtype.append(dtime)
                    else:
                        changed_dtype.append(val) #This indicate the dtype is a string
                else:
                    changed_dtype.append(val) #This could count for symbols in a feature
            
            except ValueError:
                raise ValueError("DataFrame columns contain one or more DataType")
            except:
                raise Exception()

            i = i+1

        data_f = pd.concat(changed_dtype,1)

        return data_f