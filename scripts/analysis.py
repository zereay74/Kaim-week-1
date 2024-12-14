from imports import pd, plt
from dateutil import parser

def check_missing_value(data):
    missing_summary = data.isnull().sum()
    print('Missing Value Summary')
   
    return missing_summary


def headline_length_check(data, column):
    data['headline_length'] = data[column].astype(str).apply(len)
    print('Headline Length Statistics')
    
    return (data['headline_length'].describe())

def count_and_sort(data, column):
    return data[column].value_counts().sort_values()

# convert date
def convert_date(data, column):
    def parse_mixed_dates(date_str):
        try:
            # Use dateutil parser for flexible parsing
            dt = parser.parse(date_str)
            # Localize naive datetimes to UTC
            if dt.tzinfo is None:
                return dt.replace(tzinfo=pd.Timestamp(0).tzinfo)
            return dt
        except Exception:
            return pd.NaT  # Return NaT for invalid dates

    # Apply the custom parser to the 'date' column
    data[column] = data[column].apply(parse_mixed_dates)
    # Ensure all dates are datetime objects and convert to UTC
    data[column] = pd.to_datetime(data[column], utc=True)
    return data.head(12)
    

def extract_date(data, column):
    
    # Extract time features
    data['year'] = data[column].dt.year
    data['month'] = data[column].dt.month
    data['day'] = data[column].dt.day
    data['day_of_week'] = data[column].dt.day_name()

    # Return the extracted features
    return data[[column, 'year', 'month', 'day', 'day_of_week']].head()


