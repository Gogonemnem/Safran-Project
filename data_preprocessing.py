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


ANOMALY_SUBCATEGORIES = [
    "Aircraft Equipment Critical",
    "Aircraft Equipment Less Severe",
    "Airspace Violation All Types",
    "ATC Issue All Types", # "ATC Issue All Types",
    "Flight Deck / Cabin / Aircraft Event Illness / Injury",
    "Flight Deck / Cabin / Aircraft Event Passenger Electronic Device",
    "Flight Deck / Cabin / Aircraft Event Passenger Misconduct",
    "Flight Deck / Cabin / Aircraft Event Smoke / Fire / Fumes / Odor",
    "Flight Deck / Cabin / Aircraft Event Other / Unknown",
    "Conflict NMAC",
    "Conflict Airborne Conflict",
    "Conflict Ground Conflict" # Not in Documentation
    # "Conflict Ground Conflict, Critical",
    # "Conflict Ground Conflict, Less Severe",
    "Deviation - Altitude Crossing Restriction Not Met",
    "Deviation - Altitude Excursion From Assigned Altitude", # "...from..."
    "Deviation - Altitude Overshoot",
    "Deviation - Altitude Undershoot",
    "Deviation - Speed All Types",
    "Deviation - Track / Heading All Types",
    "Deviation / Discrepancy - Procedural Clearance",
    "Deviation / Discrepancy - Procedural FAR",
    "Deviation / Discrepancy - Procedural Hazardous Material Violation",
    "Deviation / Discrepancy - Procedural Landing Without Clearance", # "...without..."
    "Deviation / Discrepancy - Procedural Maintenance",
    "Deviation / Discrepancy - Procedural MEL / CDL",
    "Deviation / Discrepancy - Procedural Published Material / Policy",
    "Deviation / Discrepancy - Procedural Security",
    "Deviation / Discrepancy - Procedural Unauthorized Flight Operations (UAS)",
    "Deviation / Discrepancy - Procedural Weight and Balance",
    "Deviation / Discrepancy - Procedural Other / Unknown",
    "Ground Excursion Ramp",
    "Ground Excursion Runway",
    "Ground Excursion Taxiway",
    "Ground Incursion Ramp",
    "Ground Incursion Runway",
    "Ground Incursion Taxiway",
    "Ground Event / Encounter Aircraft",
    "Ground Event / Encounter FOD",
    "Ground Event / Encounter Fuel Issue",
    "Ground Event / Encounter Gear Up Landing",
    "Ground Event / Encounter Ground Equipment Issue",
    "Ground Event / Encounter Ground Strike - Aircraft",
    "Ground Event / Encounter Jet Blast",
    "Ground Event / Encounter Loss Of Aircraft Control", # "...of..."
    # "Ground Event / Encounter Loss of VLOS (UAS)",
    "Ground Event / Encounter Object",
    "Ground Event / Encounter Person / Animal / Bird",
    "Ground Event / Encounter Vehicle",
    "Ground Event / Encounter Weather / Turbulence",
    "Ground Event / Encounter Other / Unknown",
    "Inflight Event / Encounter Aircraft",
    "Inflight Event / Encounter CFTT / CFIT",
    # "Inflight Event / Encounter Fly Away (UAS)",
    "Inflight Event / Encounter Fuel Issue",
    "Inflight Event / Encounter Laser",
    "Inflight Event / Encounter Loss of Aircraft Control",
    "Inflight Event / Encounter Object",
    "Inflight Event / Encounter Bird / Animal",
    "Inflight Event / Encounter Unstabilized Approach",
    "Inflight Event / Encounter VFR In IMC", # '...in...'
    "Inflight Event / Encounter Wake Vortex Encounter",
    "Inflight Event / Encounter Weather / Turbulence",
    "Inflight Event / Encounter Other / Unknown",
    "No Specific Anomaly Occurred All Types",
    "No Specific Anomaly Occurred Unwanted Situation",
]
