# AlzPy: A Python Library for Clinical Data Processing and Analysis

For the sequential processing and statistical analysis of clinical and imaging data, designed to create reproducible research pipelines.

### Scientific and Technical Utility

Streamlines the preprocessing of complex, multi-modal clinical datasets for two specific types of statistical analysis:

1.  **Cross-sectional group comparison**: Between two groups, the library performs:
      * **Welch's t-test** for continuous variables (`scipy.stats.ttest_ind` with `equal_var=False`).
      * **Pearson's Chi-squared test** for categorical variables (`scipy.stats.chi2_contingency`).
2.  **Longitudinal or nested data analysis**: The library fits **Linear Mixed-Effects Models** (`statsmodels.regression.mixed_linear_model.MixedLM`) to account for non-independence in data (e.g., repeated measures from a single subject).

Creates a clean, imputed DataFrame structured for these specific statistical tests.

-----

### Scientific Advantages and Limitations

#### Advantages

  * **Appropriate for Nested Data**: Models non-independent data structures (e.g., multiple scans from one patient) using Linear Mixed-Effects Models, a critical requirement for many clinical study designs.
  * **Reproducible Workflow**: Encapsulates data processing and analysis in a scriptable pipeline, enhancing the reproducibility and transparency of the research.
  * **Systematic Data Handling**: A structured approach to data cleaning and filtering based on pre-defined parameters, reducing the potential for subjective bias in data selection.
  * **Advanced Imputation**: Multivariate imputation (`IterativeImputer`) to handle missing data, which is generally more robust than simpler methods like mean or median imputation.

#### Limitations

  * **Linearity Assumption**: Linear Mixed-Effects Model assumes that the relationship between predictors and the outcome is linear. This may not be appropriate for all biological phenomena and could lead to incorrect conclusions if the true relationship is non-linear.
  * **Binary Group Comparison Only**: `analyze_groups` function is strictly for two-group comparisons and cannot be used for analyses involving more than two groups (e.g., ANOVA-like analyses).
  * **Imputation Assumptions**: Imputation methods make assumptions about the mechanism of missingness. The implemented imputer may introduce bias if data are missing not at random (MNAR).
  * **Model Specification Dependence**: Validity of the mixed-effects model is highly dependent on the user's correct specification of fixed and random effects. The library provides the tool, but the scientific validity of the model rests on the user's expertise.

-----

## Components

| Component         | File/Directory                  | Description                                                                 |
| :---------------- | :------------------------------ | :-------------------------------------------------------------------------- |
| **Data Processing** | `AlzPy/data_processing.py`      | Loading, merging, filtering, cleaning, and imputing data. |
| **Data Analysis** | `AlzPy/data_analysis.py`        | Group comparison and mixed-effects modeling.          |
| **Package Init** | `AlzPy/__init__.py`             | Exposes module functions under the `alzpy` namespace for direct import.      |
| **Installation** | `setup.py`, `requirements.txt`  | Package metadata and dependencies for `pip` installation.                  |

-----

## Instructions

1.  **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

2.  **Install Library**:

    ```bash
    python setup.py install
    ```

3.  **Import Functions**:

    ```python
    from alzpy import (
        load_and_merge_data,
        drop_missing_data,
        filter_scans_by_parameters,
        select_earliest_scans,
        impute_missing_data,
        analyze_groups,
        mixed_effects_analysis
    )
    ```

4.  **Example Workflow**:

    ```python
    # Load and process data
    df = load_and_merge_data(
        population_data_path='path/to/population_data.csv',
        tabs_automorph_path='path/to/imaging_data.csv'
    )
    df_clean = drop_missing_data(df)
    df_filtered = filter_scans_by_parameters(df_clean)
    df_selected = select_earliest_scans(df_filtered)
    df_imputed = impute_missing_data(
        df_selected,
        columns_to_impute=['age', 'cognitive_score']
    )

    # Run analyses
    group_results = analyze_groups(
        df_imputed,
        cat_columns=['sex', 'APOE4_status'],
        con_columns=['age', 'education_years'],
        column_of_interest='diagnosis_group'
    )
    print(group_results)

    model_results = mixed_effects_analysis(
        df_imputed,
        dependent_variables=['hippocampal_volume'],
        feature_of_interest='treatment_status',
        confounding_variables=['age', 'sex'],
        group_variable='subject_id'
    )
    print(model_results)
    ```
