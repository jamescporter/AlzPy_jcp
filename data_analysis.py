import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
from patsy import dmatrices

def analyze_groups(df, cat_columns, con_columns, column_of_interest):
    """
    Performs descriptive analysis between two groups on categorical and continuous variables.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        cat_columns (list): List of categorical columns to analyze.
        con_columns (list): List of continuous columns to analyze.
        column_of_interest (str): The column defining the groups (binary 0/1).
    
    Returns:
        pd.DataFrame: A DataFrame containing the analysis results.
    
    Raises:
        KeyError: If specified columns are not in the DataFrame.
        ValueError: If the column_of_interest does not contain exactly two unique values.
    """
    # Check if all specified columns exist in the DataFrame
    required_columns = cat_columns + con_columns + [column_of_interest]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Columns not found in DataFrame: {', '.join(missing_cols)}")
    
    # Check if column_of_interest contains exactly two groups
    unique_groups = df[column_of_interest].dropna().unique()
    if len(unique_groups) != 2:
        raise ValueError(f"The column_of_interest '{column_of_interest}' must contain exactly two unique values.")
    
    # Initialize the output DataFrame
    results_df = pd.DataFrame(columns=['Characteristic', 'Group 1', 'Group 2', 'P Value'])
    
    # Define group labels
    group_labels = unique_groups
    group1_df = df[df[column_of_interest] == group_labels[0]]
    group2_df = df[df[column_of_interest] == group_labels[1]]
    
    # Analyze continuous variables
    for con_col in con_columns:
        # Drop NaN values for the current column
        group1_data = group1_df[con_col].dropna()
        group2_data = group2_df[con_col].dropna()
        
        # Calculate means and standard errors
        mean_group1 = group1_data.mean()
        mean_group2 = group2_data.mean()
        std_err_group1 = stats.sem(group1_data)
        std_err_group2 = stats.sem(group2_data)
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=False, nan_policy='omit')
        
        # Append the results
        results_df = results_df.append({
            'Characteristic': con_col,
            'Group 1': f"{mean_group1:.2f} ± {std_err_group1:.2f}",
            'Group 2': f"{mean_group2:.2f} ± {std_err_group2:.2f}",
            'P Value': f"{p_value:.4f}"
        }, ignore_index=True)
    
    # Analyze categorical variables
    for cat_col in cat_columns:
        # Cross-tabulation
        contingency_table = pd.crosstab(df[cat_col], df[column_of_interest])
        
        # Perform Chi-squared test
        chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        # Append the overall result for the categorical variable
        results_df = results_df.append({
            'Characteristic': cat_col,
            'Group 1': '',
            'Group 2': '',
            'P Value': f"{p_value:.4f}"
        }, ignore_index=True)
        
        # Get the categories present in the variable
        categories = contingency_table.index
        for category in categories:
            count_group1 = contingency_table.loc[category, group_labels[0]] if group_labels[0] in contingency_table.columns else 0
            count_group2 = contingency_table.loc[category, group_labels[1]] if group_labels[1] in contingency_table.columns else 0
            prop_group1 = count_group1 / group1_df.shape[0] if group1_df.shape[0] > 0 else 0
            prop_group2 = count_group2 / group2_df.shape[0] if group2_df.shape[0] > 0 else 0
            
            # Append category-specific results
            results_df = results_df.append({
                'Characteristic': f"  {category}",
                'Group 1': f"{prop_group1:.2f} ({count_group1})",
                'Group 2': f"{prop_group2:.2f} ({count_group2})",
                'P Value': ''
            }, ignore_index=True)
    
    return results_df

def mixed_effects_analysis(
    df,
    dependent_variables,
    feature_of_interest,
    confounding_variables,
    group_variable,
    categorical_variables=None,
    scale_variables=None,
    verbose=True
):
    # Check if all specified columns exist in the DataFrame
    all_vars = dependent_variables + [feature_of_interest] + confounding_variables + [group_variable]
    if categorical_variables:
        all_vars += categorical_variables
    if scale_variables:
        all_vars += list(scale_variables.keys())
    missing_cols = [col for col in all_vars if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Columns not found in DataFrame: {', '.join(missing_cols)}")
    
    # Ensure group variable is treated as categorical
    df[group_variable] = df[group_variable].astype('category')
    
    # Initialize an empty DataFrame to store results
    results_df = pd.DataFrame(columns=['Dependent Variable', 'Coefficient', 'StdErr', 'z-Value', 'P-Value', 'CI_Lower', 'CI_Upper'])
    
    # Loop over each dependent variable
    for dependent_var in dependent_variables:
        # Construct the formula
        # Start with the dependent variable and feature of interest
        formula = f"{dependent_var} ~ {feature_of_interest}"
        
        # Add confounding variables
        if confounding_variables:
            conf_vars_str = ' + '.join(confounding_variables)
            formula += ' + ' + conf_vars_str
        
        # Apply scaling to variables if specified
        if scale_variables:
            for var, expression in scale_variables.items():
                formula = formula.replace(var, expression)
        
        # Handle categorical variables
        if categorical_variables:
            for cat_var in categorical_variables:
                formula = formula.replace(cat_var, f'C({cat_var})')
        
        # Fit the mixed-effects model
        try:
            model = MixedLM.from_formula(formula, groups=df[group_variable], data=df)
            result = model.fit()
        except Exception as e:
            if verbose:
                print(f"Error fitting model for {dependent_var}: {e}")
            continue  # Skip to the next dependent variable
        
        # Extract the results for the feature of interest
        if feature_of_interest in result.params.index:
            coef = result.params[feature_of_interest]
            stderr = result.bse[feature_of_interest]
            z_value = result.tvalues[feature_of_interest]
            p_value = result.pvalues[feature_of_interest]
            ci_lower, ci_upper = result.conf_int().loc[feature_of_interest]
        else:
            # If the feature of interest is not in the model (e.g., due to collinearity), skip
            if verbose:
                print(f"Feature '{feature_of_interest}' not in model for {dependent_var}")
            continue
        
        # Store the results in the DataFrame
        results_df = results_df.append({
            'Dependent Variable': dependent_var,
            'Coefficient': coef,
            'StdErr': stderr,
            'z-Value': z_value,
            'P-Value': p_value,
            'CI_Lower': ci_lower,
            'CI_Upper': ci_upper
        }, ignore_index=True)
        
        if verbose:
            print(f"Completed analysis for {dependent_var}")
    
    return results_df