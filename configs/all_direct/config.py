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

    ext_type = 'MW'

    # Seasons used

    seasons_train = [42, 90]
    seasons_test  = [108]

    # Use a subset only ofr checking

    fast = True