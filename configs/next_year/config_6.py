"""
    Partition parameter (ages, seasons)
    Extension types (MW, OM, sm, PCA ...)
"""

class configuration:

    # Lattice Lipschitz general or by ages

    divide_ages     =   True
    ages_division   =   [22, 25, 28, 32]

    # Train - Test division
    # Train: season 2019/2020 -> 2020/2021 (season_id = 42)
    # Test:  season 2020/2021 -> 2021/2022 (season_id = 90)

    seasons_train = [42]
    seasons_test  = [90]

    # Type of extension computed

    ext_type = "OM"

    # Use of PCA (all: next year + value pred., first: only next year)

    PCA = False

    # Fast used olny for checking code

    fast = False