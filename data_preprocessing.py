import pandas as pd

def load_data(path, labels, add_other=False, pp_path=None):
    loaded_data = pd.read_pickle(path)[0]

    # Drop Anomaly NaN's
    loaded_data = loaded_data.dropna(subset=['Anomaly'])#.reset_index(drop=True)

    # Convert the 'Anomaly' column to a list of lists
    anomaly_series = loaded_data['Anomaly']
    anomaly_list = anomaly_series.str.split(';').apply(lambda x: [item.strip() for item in x])

    # Initialize a DataFrame to hold the one-hot-encoded anomalies
    anomaly_df = pd.DataFrame(index=loaded_data.index)

    # Populate the DataFrame with one-hot-encoded columns for each prefix
    for prefix in labels:
        anomaly_df[prefix] = anomaly_list.apply(lambda anomalies: any(anomaly.startswith(prefix) for anomaly in anomalies)).astype(int)

    # Add the 'Other' category
    if add_other:
        anomaly_df['Other'] = (anomaly_df.sum(axis=1) == 0).astype(int)

    # Assign the one-hot-encoded anomalies as a new column 'labels' to 'loaded_data'
    loaded_data['labels'] = anomaly_df.apply(lambda row: row.tolist(), axis=1)

    # Now, 'loaded_data' is a DataFrame that includes both the 'text' and 'labels' columns
    if pp_path is None:
        loaded_data['text'] = loaded_data["Narrative"]
    else:
        loaded_data['text'] = pd.read_pickle(pp_path)

    # If you want to create a new DataFrame with just 'text' and 'labels':
    final_df = loaded_data[['text', 'labels']]
    return final_df


# Root label (source = ASRS coding forms) : order = by descending frequency
ANOMALY_LABELS = ['Deviation / Discrepancy - Procedural',
                    'Aircraft Equipment',
                    'Conflict',
                    'Inflight Event / Encounter',
                    'ATC Issue',
                    'Deviation - Altitude',
                    'Deviation - Track / Heading',
                    'Ground Event / Encounter',
                    'Flight Deck / Cabin / Aircraft Event',
                    'Ground Incursion',
                    'Airspace Violation',
                    'Deviation - Speed',
                    'Ground Excursion',
                    'No Specific Anomaly Occurred']


