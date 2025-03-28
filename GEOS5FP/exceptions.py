class GEOS5FPGranuleNotAvailable(Exception):
    pass


class GEOS5FPDayNotAvailable(Exception):
    pass


class GEOS5FPMonthNotAvailable(Exception):
    pass


class GEOS5FPYearNotAvailable(Exception):
    pass


class FailedGEOS5FPDownload(ConnectionError):
    pass
