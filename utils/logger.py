import logging

def get_logger(logfile):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(logfile, mode='a')
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(filename)s[Line:%(lineno)s] - %(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger