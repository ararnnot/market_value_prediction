"""
    All: next_year + market value
"""

class configuration:

    # Used only for importing the same training - test data
    config_next_year    = 1
    config_market_value = 1

    # Last season (from 2)
    seasons_train = [90]
    seasons_test  = [108]

    # Network architecture

    fun_act  = "relu"
    h_layers = [1024]

    # Use a subset only ofr checking

    fast = False