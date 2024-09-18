import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder

def load_and_merge_data(
    population_data=None,
    tabs_automorph=None,
    population_data_path=None,
    tabs_automorph_path=None,
    left_key='dcm_id',
    right_key='patient_id_tabs'
):
    """
    Loads population data and tabs automorph data from DataFrames or CSV files and merges them.

    Parameters:
        population_data (pd.DataFrame, optional): DataFrame containing the population data.
        tabs_automorph (pd.DataFrame, optional): DataFrame containing the tabs automorph data.
        population_data_path (str, optional): File path to the population data CSV file.
        tabs_automorph_path (str, optional): File path to the tabs automorph CSV file.
        left_key (str, optional): Column name to merge on from the population data (default 'dcm_id').
        right_key (str, optional): Column name to merge on from the tabs automorph data (default 'patient_id_tabs').

    Returns:
        pd.DataFrame: A DataFrame resulting from merging the population and tabs automorph data.

    Raises:
        ValueError: If neither DataFrame nor file path is provided for either dataset.
        FileNotFoundError: If a specified file path does not exist.
        pd.errors.EmptyDataError: If a specified file is empty.
        KeyError: If the specified merge keys are not found in the DataFrames.
    """

    # Load population data
    if population_data is None:
        if population_data_path is None:
            raise ValueError("Either 'population_data' DataFrame or 'population_data_path' must be provided.")
        else:
            try:
                population_data = pd.read_csv(population_data_path)
            except FileNotFoundError:
                raise FileNotFoundError(f"Population data file not found at '{population_data_path}'.")
            except pd.errors.EmptyDataError:
                raise pd.errors.EmptyDataError(f"Population data file at '{population_data_path}' is empty.")

    # Load tabs automorph data
    if tabs_automorph is None:
        if tabs_automorph_path is None:
            raise ValueError("Either 'tabs_automorph' DataFrame or 'tabs_automorph_path' must be provided.")
        else:
            try:
                tabs_automorph = pd.read_csv(tabs_automorph_path)
            except FileNotFoundError:
                raise FileNotFoundError(f"Tabs automorph data file not found at '{tabs_automorph_path}'.")
            except pd.errors.EmptyDataError:
                raise pd.errors.EmptyDataError(f"Tabs automorph data file at '{tabs_automorph_path}' is empty.")

    # Validate merge keys
    if left_key not in population_data.columns:
        raise KeyError(f"Merge key '{left_key}' not found in population data columns.")
    if right_key not in tabs_automorph.columns:
        raise KeyError(f"Merge key '{right_key}' not found in tabs automorph data columns.")

    # Merge the DataFrames
    merged_df = pd.merge(
        population_data,
        tabs_automorph,
        left_on=left_key,
        right_on=right_key,
        how='inner'
    )

    return merged_df


def get_subject_and_scan_counts(df, subject_id_col='dcm_id', return_obj=False):
    """
    Calculates and prints the total number of unique subjects and total number of scans in the DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        subject_id_col (str, optional): The name of the column representing subject IDs.
                                        Defaults to 'dcm_id'.
        return_obj (bool, optional): If True, returns the counts as variables.
                                     If False, only prints the counts. Defaults to False.

    Returns:
        tuple or None:
            - If return_obj is True: Returns a tuple containing (total_subjects, total_scans).
            - If return_obj is False: Returns None.

    Raises:
        KeyError: If the specified subject_id_col is not in the DataFrame columns.
    """
    if subject_id_col not in df.columns:
        raise KeyError(f"Column '{subject_id_col}' not found in the DataFrame.")

    total_subjects = df[subject_id_col].nunique()
    total_scans = df.shape[0]

    print('Total Number of Subjects:', total_subjects)
    print('Total Number of Scans:', total_scans)

    if return_obj:
        return total_subjects, total_scans
    

def drop_missing_data(df, date_col='date', image_id_col='image_id', verbose=True):
    """
    Drops rows from the DataFrame where the date or image_id columns have missing values.

    Parameters:
        df (pd.DataFrame): The DataFrame to clean.
        date_col (str, optional): The name of the date column. Defaults to 'date'.
        image_id_col (str, optional): The name of the image ID column. Defaults to 'image_id'.
        verbose (bool, optional): If True, prints the number of rows after each operation. Defaults to True.

    Returns:
        pd.DataFrame: The cleaned DataFrame.

    Raises:
        KeyError: If the specified date_col or image_id_col is not in the DataFrame columns.
    """
    # Check if the specified columns exist
    missing_cols = [col for col in [date_col, image_id_col] if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Columns not found in DataFrame: {', '.join(missing_cols)}")

    # Drop rows with NaN in date column
    df_cleaned = df.dropna(subset=[date_col])
    if verbose:
        print(f"Number of Rows after dropping NaN in '{date_col}': {df_cleaned.shape[0]}")

    # Drop rows with NaN in image_id column
    df_cleaned = df_cleaned.dropna(subset=[image_id_col])
    if verbose:
        print(f"Number of Rows after dropping NaN in '{image_id_col}': {df_cleaned.shape[0]}")

    return df_cleaned

def filter_scans_by_parameters(
    df,
    fixation_col='fixation',
    fixation_value='Macula',
    scan_protocol_col='scan_protocol',
    scan_protocol_value='Macula',
    n_bscan_col='n_bscan',
    n_bscan_value=128,
    verbose=True
):
    """
    Filters the DataFrame to include only scans that meet specific technical parameters.
    Also provides counts of scans removed due to not meeting each parameter.

    Parameters:
        df (pd.DataFrame): The DataFrame to filter.
        fixation_col (str, optional): The name of the fixation column. Defaults to 'fixation'.
        fixation_value (str, optional): The required value in the fixation column. Defaults to 'Macula'.
        scan_protocol_col (str, optional): The name of the scan protocol column. Defaults to 'scan_protocol'.
        scan_protocol_value (str, optional): The required value in the scan protocol column. Defaults to 'Macula'.
        n_bscan_col (str, optional): The name of the n_bscan column. Defaults to 'n_bscan'.
        n_bscan_value (int, optional): The required value in the n_bscan column. Defaults to 128.
        verbose (bool, optional): If True, prints detailed information about scans removed. Defaults to True.

    Returns:
        pd.DataFrame: The filtered DataFrame.

    Raises:
        KeyError: If any of the specified columns are not in the DataFrame.
    """
    # Check if the specified columns exist
    required_columns = [fixation_col, scan_protocol_col, n_bscan_col]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Columns not found in DataFrame: {', '.join(missing_cols)}")

    total_scans_before = df.shape[0]

    # Create masks for each condition
    mask_fixation = df[fixation_col] == fixation_value
    mask_scan_protocol = df[scan_protocol_col] == scan_protocol_value
    mask_n_bscan = df[n_bscan_col] == n_bscan_value

    # Count scans that do not meet each criterion
    scans_wrong_fixation = total_scans_before - mask_fixation.sum()
    scans_wrong_scan_protocol = total_scans_before - mask_scan_protocol.sum()
    scans_wrong_n_bscan = total_scans_before - mask_n_bscan.sum()

    # Apply all filters together
    df_filtered = df[mask_fixation & mask_scan_protocol & mask_n_bscan]
    total_scans_after = df_filtered.shape[0]

    if verbose:
        print(f"Total scans before filtering: {total_scans_before}")
        print(f"Scans removed due to incorrect '{fixation_col}': {scans_wrong_fixation}")
        print(f"Scans removed due to incorrect '{scan_protocol_col}': {scans_wrong_scan_protocol}")
        print(f"Scans removed due to incorrect '{n_bscan_col}': {scans_wrong_n_bscan}")
        print(f"Total scans after filtering: {total_scans_after}")

    return df_filtered

def filter_scans_by_quality(
    df,
    quantile_columns=None,
    verbose=True
):
    """
    Filters the DataFrame by applying quantile-based thresholds on specified columns.

    Parameters:
        df (pd.DataFrame): The DataFrame to filter.
        quantile_columns (dict, optional): A dictionary where keys are column names and values are quantile thresholds.
            Example:
            {
                'ilm_indicator': 0.2,
                'min_motion_correlation': 0.2,
                'max_motion_delta': 0.8,
                'max_motion_factor': 0.8,
                'quality': 0.2
            }
            Defaults to the specified columns and quantiles if not provided.
        verbose (bool, optional): If True, prints the number of rows after filtering. Defaults to True.

    Returns:
        pd.DataFrame: The filtered DataFrame.

    Raises:
        KeyError: If any of the specified columns are not in the DataFrame.
        ValueError: If quantile thresholds are not between 0 and 1.
    """
    # Default quantile columns and thresholds
    if quantile_columns is None:
        quantile_columns = {
            'ilm_indicator': 0.2,
            'min_motion_correlation': 0.2,
            'max_motion_delta': 0.8,
            'max_motion_factor': 0.8,
            'quality': 0.2
        }

    # Validate quantile thresholds
    for col, quantile in quantile_columns.items():
        if not 0 <= quantile <= 1:
            raise ValueError(f"Quantile threshold for '{col}' must be between 0 and 1.")

    # Check if the specified columns exist
    missing_cols = [col for col in quantile_columns.keys() if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Columns not found in DataFrame: {', '.join(missing_cols)}")

    # Calculate quantiles
    quantile_values = {}
    for col, quantile in quantile_columns.items():
        quantile_values[col] = df[col].quantile(quantile)

    # Build filtering conditions
    conditions = []
    for col, quantile in quantile_columns.items():
        threshold = quantile_values[col]
        if quantile <= 0.5:
            # For lower quantiles, we filter for values >= threshold
            condition = df[col] >= threshold
        else:
            # For upper quantiles, we filter for values <= threshold
            condition = df[col] <= threshold
        conditions.append(condition)

    # Combine all conditions using logical AND
    combined_condition = conditions[0]
    for condition in conditions[1:]:
        combined_condition &= condition

    # Apply the filter
    df_filtered = df[combined_condition]

    if verbose:
        print(f"Number of Rows after applying quality filters: {df_filtered.shape[0]}")

    return df_filtered

def filter_scans_by_prediction_qa(
    df,
    prediction_qa_col='Prediction_qa',
    exclude_value=2,
    verbose=True
):
    """
    Filters the DataFrame by excluding rows where the prediction QA column equals a specified value.

    Parameters:
        df (pd.DataFrame): The DataFrame to filter.
        prediction_qa_col (str, optional): The name of the prediction QA column. Defaults to 'Prediction_qa'.
        exclude_value (int or float, optional): The value to exclude from the DataFrame. Defaults to 2.
        verbose (bool, optional): If True, prints the number of rows after filtering. Defaults to True.

    Returns:
        pd.DataFrame: The filtered DataFrame.

    Raises:
        KeyError: If the specified prediction_qa_col is not in the DataFrame.
    """
    # Check if the specified column exists
    if prediction_qa_col not in df.columns:
        raise KeyError(f"Column '{prediction_qa_col}' not found in the DataFrame.")

    total_rows_before = df.shape[0]

    # Filter the DataFrame
    df_filtered = df[df[prediction_qa_col] != exclude_value]
    total_rows_after = df_filtered.shape[0]

    if verbose:
        print(f"Number of rows before filtering: {total_rows_before}")
        print(f"Number of rows after filtering: {total_rows_after}")
        print(f"Number of rows removed: {total_rows_before - total_rows_after}")

    return df_filtered

def remove_duplicates(
    df,
    duplicate_col='image_id',
    verbose=True
):
    """
    Removes duplicates from the DataFrame based on a specified column by keeping the row with the fewest NaN values.
    If multiple rows have the same minimal number of NaN values, one is selected at random.

    Parameters:
        df (pd.DataFrame): The DataFrame to process.
        duplicate_col (str, optional): The column name to identify duplicates. Defaults to 'image_id'.
        verbose (bool, optional): If True, prints the number of rows after duplicates are removed. Defaults to True.

    Returns:
        pd.DataFrame: The DataFrame after processing duplicates.

    Raises:
        KeyError: If the specified duplicate_col is not in the DataFrame.
    """
    # Check if the specified column exists
    if duplicate_col not in df.columns:
        raise KeyError(f"Column '{duplicate_col}' not found in the DataFrame.")

    total_rows_before = df.shape[0]

    # Define the function to process duplicates
    def process_duplicates(sub_df):
        # Calculate the number of NaNs in each row
        nan_counts = sub_df.isna().sum(axis=1)
        min_nan_count = nan_counts.min()

        # Filter rows with the minimum NaN count
        candidates = sub_df[nan_counts == min_nan_count]

        # If more than one row remains, randomly select one
        if len(candidates) > 1:
            return candidates.sample(n=1)
        else:
            return candidates

    # Identify all duplicates based on the duplicate column
    duplicates = df[df.duplicated(duplicate_col, keep=False)]

    # Process duplicates
    processed_duplicates = (
        duplicates.groupby(duplicate_col, as_index=False, group_keys=False)
        .apply(process_duplicates)
    )

    # Get all unique rows that were not part of duplicates
    unique_rows = df.drop_duplicates(duplicate_col, keep=False)

    # Combine the unique rows with the processed duplicates
    df_processed = pd.concat([unique_rows, processed_duplicates], ignore_index=True)

    total_rows_after = df_processed.shape[0]

    if verbose:
        print(f"Number of rows before removing duplicates: {total_rows_before}")
        print(f"Number of rows after removing duplicates: {total_rows_after}")
        print(f"Number of duplicates removed: {total_rows_before - total_rows_after}")

    return df_processed


def select_earliest_scans(
    df,
    patient_id_col='patient_id_tabs',
    eye_col='eye',
    date_col='date',
    softmax_good_col='softmax_good_mean_qa',
    softmax_usable_col='softmax_usable_mean_qa',
    verbose=True
):
    """
    Selects the earliest pair of scans (left and right eye) for each subject.

    Parameters:
        df (pd.DataFrame): The DataFrame to process.
        patient_id_col (str, optional): Column name for patient IDs. Defaults to 'patient_id_tabs'.
        eye_col (str, optional): Column name for eye information. Defaults to 'eye'.
        date_col (str, optional): Column name for scan dates. Defaults to 'date'.
        softmax_good_col (str, optional): Column name for 'softmax_good_mean_qa'. Defaults to 'softmax_good_mean_qa'.
        softmax_usable_col (str, optional): Column name for 'softmax_usable_mean_qa'. Defaults to 'softmax_usable_mean_qa'.
        verbose (bool, optional): If True, prints the number of images after processing. Defaults to True.

    Returns:
        pd.DataFrame: DataFrame where each subject has one left and one right eye scan.

    Raises:
        KeyError: If any of the specified columns are not in the DataFrame.
    """
    # Check if the specified columns exist
    required_columns = [
        patient_id_col,
        eye_col,
        date_col,
        softmax_good_col,
        softmax_usable_col
    ]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Columns not found in DataFrame: {', '.join(missing_cols)}")

    # Generate 'new_softmax' column
    df['new_softmax'] = df[softmax_good_col] + df[softmax_usable_col]

    # Convert 'date' column to datetime
    df[date_col] = pd.to_datetime(df[date_col])

    # Sort the DataFrame
    df_sorted = df.sort_values(
        by=[patient_id_col, eye_col, date_col, 'new_softmax'],
        ascending=[True, True, True, False]  # For 'new_softmax', higher values are better
    )

    # Drop duplicates to keep the earliest scan per eye per patient
    df_earliest = df_sorted.drop_duplicates(subset=[patient_id_col, eye_col], keep='first')

    # Filter patients who have scans for both eyes
    patient_counts = df_earliest[patient_id_col].value_counts()
    valid_patients = patient_counts[patient_counts == 2].index

    df_final = df_earliest[df_earliest[patient_id_col].isin(valid_patients)].copy()

    if verbose:
        print(f"Number of images after selecting earliest scans per eye per subject: {df_final.shape[0]}")

    return df_final

def impute_missing_data(
    df,
    columns_to_impute,
    categorical_columns=None,
    max_iter=10,
    random_state=None,
    sample_posterior=False,
    min_value=None,
    max_value=None,
    verbose=True
):
    """
    Imputes missing values in the DataFrame using Iterative Imputer.

    Parameters:
        df (pd.DataFrame): The DataFrame containing data to impute.
        columns_to_impute (list): List of column names to impute.
        categorical_columns (list, optional): List of categorical columns among the columns to impute.
            Defaults to None.
        max_iter (int, optional): Maximum number of imputation iterations. Defaults to 10.
        random_state (int, optional): Random seed for reproducibility. Defaults to None.
        sample_posterior (bool, optional): Whether to sample from the posterior distribution of each imputation.
            Defaults to False.
        min_value (float, optional): Minimum possible imputed value. Defaults to None.
        max_value (float, optional): Maximum possible imputed value. Defaults to None.
        verbose (bool, optional): If True, prints the number of missing values before and after imputation.
            Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame with imputed values.

    Raises:
        KeyError: If specified columns are not in the DataFrame.
    """
    # Check if specified columns exist in the DataFrame
    missing_cols = [col for col in columns_to_impute if col not in df.columns]
    if missing_cols:
        raise KeyError(f"The following columns are not in the DataFrame: {', '.join(missing_cols)}")

    # If categorical_columns is None, assume no columns are categorical
    if categorical_columns is None:
        categorical_columns = []
    else:
        # Check if categorical_columns are in columns_to_impute
        invalid_cats = [col for col in categorical_columns if col not in columns_to_impute]
        if invalid_cats:
            raise KeyError(f"The following categorical columns are not in columns_to_impute: {', '.join(invalid_cats)}")

    df_impute = df[columns_to_impute].copy()

    # Initialize OrdinalEncoder for categorical columns
    encoders = {}
    for col in categorical_columns:
        encoder = OrdinalEncoder()
        df_impute[[col]] = encoder.fit_transform(df_impute[[col]])
        encoders[col] = encoder

    # Initialize the IterativeImputer with additional parameters
    imputer = IterativeImputer(
        max_iter=max_iter,
        random_state=random_state,
        sample_posterior=sample_posterior,
        min_value=min_value,
        max_value=max_value
    )

    # Impute the data
    imputed_array = imputer.fit_transform(df_impute)
    df_imputed = pd.DataFrame(imputed_array, columns=columns_to_impute, index=df_impute.index)

    # Round the imputed categorical columns and inverse transform
    for col in categorical_columns:
        df_imputed[col] = np.round(df_imputed[col]).astype(int)
        df_imputed[[col]] = encoders[col].inverse_transform(df_imputed[[col]])

    # Replace the original columns with imputed data
    df_imputed_full = df.copy()
    df_imputed_full[columns_to_impute] = df_imputed[columns_to_impute]

    if verbose:
        missing_before = df[columns_to_impute].isnull().sum().sum()
        missing_after = df_imputed[columns_to_impute].isnull().sum().sum()
        print(f"Total missing values before imputation: {missing_before}")
        print(f"Total missing values after imputation: {missing_after}")

    return df_imputed_full