def compute_expected_final_nb_points(lin_coeff, nb_pts_at_break, breaking_leg, final_leg, basic=True):
    if basic:
        return lin_coeff*final_leg

    else:
        return nb_pts_at_break + lin_coeff*(final_leg - breaking_leg)

def basic(data_df, breaking_leg=27, final_leg=38):
    cols = ['lr_feat_coeff', 'nb_pts_at_break']

    data_df['predicted_final_nb_pts'] = data_df[cols].apply(
        lambda r: compute_expected_final_nb_points(lin_coeff=r[0],
                                                   nb_pts_at_break=r[1],
                                                   breaking_leg=breaking_leg,
                                                   final_leg=final_leg, 
                                                   basic=True), 
        axis=1)

    return data_df
