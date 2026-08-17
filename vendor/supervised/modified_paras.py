        dt_params = {
            "criterion": "friedman_mse", 
            "max_depth": 4},
        regression_rf_params = {
            "criterion": "mse",
            "max_features": 0.6,
            "min_samples_split": 30,
            "max_depth": 12},
        regression_et_params  = {
            "criterion": "mse",
            "max_features": 0.6,
            "min_samples_split": 30,
            "max_depth": 12},
        lgbm_bin_params = {
            "objective": "regression",
            "metric": "l2",
            "num_leaves": 31,
            "learning_rate": 0.1,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "min_data_in_leaf": 15},
        nn_params = {
            "dense_layers": 2,
            "dense_1_size": 64,
            "dense_2_size": 32,
            "dropout": 0,
            "learning_rate": 0.05,
            "momentum": 0.9,
            "decay": 0.001}