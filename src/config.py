# model model: True: make predictions on incomplete loans. False: Only output model metrics
generate_model_predictions = True

# list of numerical features for model training
num_feat_list = [
    "loan_amnt",
    "funded_amnt",
    "int_rate",
    "installment",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "mths_since_last_delinq",
    "open_acc",
    "mths_since_first_cr_line",
    "term",
    "revol_bal",
    "total_acc",
    "out_prncp",
]

# list of categorical features for preprocessing
cat_feat_list = [
    "emp_length",
    "home_ownership",
    "purpose",
    "addr_state",
]
