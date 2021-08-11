from tpd import recorder

LOGS_PATH = 'logs'

recorder.start(base_folder='./', timestamp=False, name=LOGS_PATH)