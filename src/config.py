LOC = 'tum'

FIGSIZE = (14,6)

BAROCORRECTED_OUTPUT_FN = '{site}_{year}_barocorrected_{segment}.csv'

NORMALIZED_FN = '{sensor_type}_{sitename}_{year}.csv'

VENTED_OUTPUT_FN = '{site}_{span}_vented_{segment}.csv'

FINAL_OUTPUT_FN = '{site}_timeseries_stage_Q_T_{start}_{end}.csv'

FINAL_OUTPUT_METADATA_FN = '{site}_timeseries_stage_Q_T_{start}_{end}_metadata.txt'

SITE_SHORTNAME = {0 : 'LyellBlwMaclure',
                  1 : 'LyellAbvTB',
                  2 : 'DanaFk@BugCamp',
                  3 : 'Tuolumne@120',
                  4 : 'BuddCreek',
                  5 : 'DelaneyAbvPCT'}

SITE_LONGNAME = {0 : 'Lyell Below Maclure',
                 1 : 'Lyell Above Twin Bridges',
                 2 : 'Dana Fork at Bug Camp',
                 3 : 'Tuolumne River at 120',
                 4 : 'Budd Creek',
                 5 : 'Delaney Above PCT'}

LEVEL_HEADER = ['lvl_cm', 'temp_C']

BARO_HEADER = ['baro_cm', 'temp_C']

BARO_CORRECTED_HEADER = ['date_time(UTC:PDT+7)', 'raw_pressure(cm)', 'barocorrected_pressure(cm)', 'water_temperature(deg_C)', 'discharge_flag']

VENTED_CORRECTED_HEADER = ['date_time(UTC:PDT+7)', 'vented_pressure(cm)', 'water_temperature(deg_C)', 'discharge_flag']

NCAR_BARO_HEADER = ['baro_cm']

FINAL_OUTPUT_HEADER = ['date_time(UTC:PDT+7)', 'raw_pressure(cm)', 'barocorrected_pressure(cm)', 'adjusted_stage(cm)', 'estimated_discharge(cms)', 'water_temperature(deg_C)', 'discharge_flag']

NCAR_TUM_LAT = 37.5 # degrees N
NCAR_TUM_LON = 240 # degrees e

ANOMOLY_FLAG = 1
ICEJAM_FLAG = 2
MALFUNCTION_FLAG = 3

RATING_CURVES = {0 : (lambda h: (5.08516 * (h - 1.9383)**1.49088 if h < 2.29893 else
                                 20.1140 * (h - 2.1167)**1.70046)), 
                 1 : (lambda h: (13.0679 * (h - 0.189547)**1.53746 if h < 0.545817 else 
                                 34.0175 * (h - 0.3297)**1.69602 if h > 0.786625 else 
                                 32.3266 * (h - 0.3171)**1.68957)), 
                 2 : (lambda h: (9.81848 * (h - 0.218727)**1.48825 if h < 0.354118 else
                                 66.8344 * (h - 0.4724)**1.68704 if h > 0.904318 else
                                 34.8425 * (h - 0.2816)**1.6152)), 
                 3 : (lambda h: (34.3522 * (h - 0.7639)**1.465 if h < 0.8011 else
                                 17.0873 * (h - 0.7076)**1.7401)), 
                 4 : None, 
                 5 : None}