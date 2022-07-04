def _get_rank_position_scoring(nb_teams: int):
    bonus_position = {1: 300,
                      2: 280,
                      3: 270,
                      4: 260,
                      5: 220,
                      nb_teams: 350,
                      nb_teams - 1: 340,
                      nb_teams - 2: 300,
                      nb_teams - 3: 280}

    for rk in range(6, nb_teams - 3):
        upper = ((nb_teams / 2) - 3) ** 2
        lower = ((nb_teams / 2) - 5) ** 2
        if 2 * rk < nb_teams:
            bonus_position[rk] = int(220 * (rk - nb_teams / 2) ** 2 / lower)
        else:
            bonus_position[rk] = int(280 * ((rk - nb_teams / 2) ** 2) / upper)

    return bonus_position


def _get_rank_scoring(nb_teams: int):
    max_bonus = 250 * nb_teams
    return {1 + i // 2 if i % 2 == 0 else nb_teams - i // 2: max_bonus - i * 250 for i in range(nb_teams)}


def get_rank_percentage_quality_dict(nb_teams: int, version: int = 1):
    if version == 1:
        return _get_rank_position_scoring(nb_teams=nb_teams)
    elif version == 2:
        return _get_rank_scoring(nb_teams=nb_teams)
    else:
        raise ValueError("Unknown version. Please enter 1 or 2 as version")
