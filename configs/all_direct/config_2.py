"""
    All: next_year + market value
"""


class configuration:

    # Distance

    normalize       =   True
    norm            =   2
    
    # Normalize

    normalize = 'sd'
    
    # Type of extension computed

    ext_type = 'OM'

    # Seasons used

    seasons_train = [42, 90]
    seasons_test  = [108]

    # Use a subset only for checking

    fast = True