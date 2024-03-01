if __name__ == "__main__":
    import itertools
    import pandas as pd

    # Define your options
    abbreviation_options = [False, True]
    loss_type_options = ['BCE', 'BinaryFocal']
    balanced_options = [False, True]
    layers_to_unfreeze_options = [None, [8, 9, 10, 11]]
    encoder_name_options = [
        'bert-base-uncased', 
        'NASA-AIML/MIKA_SafeAeroBERT', 
        'allenai/longformer-base-4096'
    ]

    batch_size_options = [32]  # Add more if needed
    epochs_options = [5]       # Add more if needed
    learning_rate_options = [1e-05 * 2]  # Add more if needed

    # Generate Cartesian product
    cartesian_product = list(itertools.product(
        abbreviation_options, 
        loss_type_options, 
        balanced_options, 
        layers_to_unfreeze_options, 
        encoder_name_options,
        batch_size_options,
        epochs_options,
        learning_rate_options
    ))

    # Determining MAX_LEN based on ENCODER_NAME
    def determine_max_len(encoder_name):
        return 1024 if encoder_name == 'allenai/longformer-base-4096' else 512

    # Creating a DataFrame with column headers
    column_names = ['Abbreviation', 'Loss Type', 'Balanced', 'Layers to Unfreeze', 'Encoder Name', 'Batch Size', 'Epochs', 'Learning Rate', 'Max Length']
    df = pd.DataFrame(cartesian_product, columns=column_names[:-1])  # Exclude 'Max Length' for now

    # Adding the 'Max Length' column based on 'Encoder Name'
    df['Max Length'] = df['Encoder Name'].apply(determine_max_len)

    # Exporting to CSV format
    csv_output = df.to_csv("model-configs.csv", index=False)

    print(pd.read_csv("model-configs.csv").head())

