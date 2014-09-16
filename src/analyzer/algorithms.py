import pandas
import numpy as np
import scipy
import statsmodels.api as sm
import traceback
import logging
from time import time
from msgpack import unpackb, packb, Unpacker
from redis import StrictRedis

import math
from os import getpid
import re

from kyotocabinet import *


from settings import (
    ALGORITHMS,
    CONSENSUS,
    FULL_DURATION,
    MAX_TOLERABLE_BOREDOM,
    MIN_TOLERABLE_LENGTH,
    STALE_PERIOD,
    REDIS_SOCKET_PATH,
    ENABLE_SECOND_ORDER,
    BOREDOM_SET_SIZE,
)

from algorithm_exceptions import *

logger = logging.getLogger("AnalyzerLog")
redis_conn = StrictRedis(unix_socket_path=REDIS_SOCKET_PATH)

"""
This is no man's land. Do anything you want in here,
as long as you return a boolean that determines whether the input
timeseries is anomalous or not.

To add an algorithm, define it here, and add its name to settings.ALGORITHMS.
"""


def tail_avg(timeseries):
    """
    This is a utility function used to calculate the average of the last three
    datapoints in the series as a measure, instead of just the last datapoint.
    It reduces noise, but it also reduces sensitivity and increases the delay
    to detection.
    """
    try:
        t = (timeseries[-1][1] + timeseries[-2][1] + timeseries[-3][1]) / 3
        return t
    except IndexError:
        return timeseries[-1][1]


def median_absolute_deviation(timeseries):
    """
    A timeseries is anomalous if the deviation of its latest datapoint with
    respect to the median is X times larger than the median of deviations.
    """

    series = pandas.Series([x[1] for x in timeseries])
    median = series.median()
    demedianed = np.abs(series - median)
    median_deviation = demedianed.median()

    # The test statistic is infinite when the median is zero,
    # so it becomes super sensitive. We play it safe and skip when this happens.
    if median_deviation == 0:
        return False

    test_statistic = demedianed.iget(-1) / median_deviation

    # Completely arbitary...triggers if the median deviation is
    # 6 times bigger than the median
    if test_statistic > 6:
        return True


def grubbs(timeseries):
    """
    A timeseries is anomalous if the Z score is greater than the Grubb's score.
    """

    series = scipy.array([x[1] for x in timeseries])
    stdDev = scipy.std(series)
    mean = np.mean(series)
    tail_average = tail_avg(timeseries)
    z_score = (tail_average - mean) / stdDev
    len_series = len(series)
    threshold = scipy.stats.t.isf(.05 / (2 * len_series), len_series - 2)
    threshold_squared = threshold * threshold
    grubbs_score = ((len_series - 1) / np.sqrt(len_series)) * np.sqrt(threshold_squared / (len_series - 2 + threshold_squared))

    return z_score > grubbs_score


def first_hour_average(timeseries):
    """
    Calcuate the simple average over one hour, FULL_DURATION seconds ago.
    A timeseries is anomalous if the average of the last three datapoints
    are outside of three standard deviations of this value.
    """
    last_hour_threshold = time() - (FULL_DURATION - 3600)
    series = pandas.Series([x[1] for x in timeseries if x[0] < last_hour_threshold])
    mean = (series).mean()
    stdDev = (series).std()
    t = tail_avg(timeseries)

    return abs(t - mean) > 3 * stdDev


def stddev_from_average(timeseries):
    """
    A timeseries is anomalous if the absolute value of the average of the latest
    three datapoint minus the moving average is greater than three standard
    deviations of the average. This does not exponentially weight the MA and so
    is better for detecting anomalies with respect to the entire series.
    """
    series = pandas.Series([x[1] for x in timeseries])
    mean = series.mean()
    stdDev = series.std()
    t = tail_avg(timeseries)

    return abs(t - mean) > 3 * stdDev


def stddev_from_moving_average(timeseries):
    """
    A timeseries is anomalous if the absolute value of the average of the latest
    three datapoint minus the moving average is greater than three standard
    deviations of the moving average. This is better for finding anomalies with
    respect to the short term trends.
    """
    series = pandas.Series([x[1] for x in timeseries])
    expAverage = pandas.stats.moments.ewma(series, com=50)
    stdDev = pandas.stats.moments.ewmstd(series, com=50)

    return abs(series.iget(-1) - expAverage.iget(-1)) > 3 * stdDev.iget(-1)


def mean_subtraction_cumulation(timeseries):
    """
    A timeseries is anomalous if the value of the next datapoint in the
    series is farther than three standard deviations out in cumulative terms
    after subtracting the mean from each data point.
    """

    series = pandas.Series([x[1] if x[1] else 0 for x in timeseries])
    series = series - series[0:len(series) - 1].mean()
    stdDev = series[0:len(series) - 1].std()
    expAverage = pandas.stats.moments.ewma(series, com=15)

    return abs(series.iget(-1)) > 3 * stdDev


def least_squares(timeseries):
    """
    A timeseries is anomalous if the average of the last three datapoints
    on a projected least squares model is greater than three sigma.
    """

    x = np.array([t[0] for t in timeseries])
    y = np.array([t[1] for t in timeseries])
    A = np.vstack([x, np.ones(len(x))]).T
    results = np.linalg.lstsq(A, y)
    residual = results[1]
    m, c = np.linalg.lstsq(A, y)[0]
    errors = []
    for i, value in enumerate(y):
        projected = m * x[i] + c
        error = value - projected
        errors.append(error)

    if len(errors) < 3:
        return False

    std_dev = scipy.std(errors)
    t = (errors[-1] + errors[-2] + errors[-3]) / 3

    return abs(t) > std_dev * 3 and round(std_dev) != 0 and round(t) != 0


def histogram_bins(timeseries):
    """
    A timeseries is anomalous if the average of the last three datapoints falls
    into a histogram bin with less than 20 other datapoints (you'll need to tweak
    that number depending on your data)

    Returns: the size of the bin which contains the tail_avg. Smaller bin size
    means more anomalous.
    """

    series = scipy.array([x[1] for x in timeseries])
    t = tail_avg(timeseries)
    h = np.histogram(series, bins=15)
    bins = h[1]
    for index, bin_size in enumerate(h[0]):
        if bin_size <= 20:
            # Is it in the first bin?
            if index == 0:
                if t <= bins[0]:
                    return True
            # Is it in the current bin?
            elif t >= bins[index] and t < bins[index + 1]:
                    return True

    return False


def ks_test(timeseries):
    """
    A timeseries is anomalous if 2 sample Kolmogorov-Smirnov test indicates
    that data distribution for last 10 minutes is different from last hour.
    It produces false positives on non-stationary series so Augmented
    Dickey-Fuller test applied to check for stationarity.
    """

    hour_ago = time() - 3600
    ten_minutes_ago = time() - 600
    reference = scipy.array([x[1] for x in timeseries if x[0] >= hour_ago and x[0] < ten_minutes_ago])
    probe = scipy.array([x[1] for x in timeseries if x[0] >= ten_minutes_ago])

    if reference.size < 20 or probe.size < 20:
        return False

    ks_d, ks_p_value = scipy.stats.ks_2samp(reference, probe)

    if ks_p_value < 0.05 and ks_d > 0.5:
        adf = sm.tsa.stattools.adfuller(reference, 10)
        if adf[1] < 0.05:
            return True

    return False


def is_anomalously_anomalous(metric_name, ensemble, datapoint):
    """
    This method runs a meta-analysis on the metric to determine whether the
    metric has a past history of triggering. TODO: weight intervals based on datapoint
    """
    # We want the datapoint to avoid triggering twice on the same data
    new_trigger = [time(), datapoint]

    # Get the old history
    raw_trigger_history = redis_conn.get('trigger_history.' + metric_name)
    if not raw_trigger_history:
        redis_conn.set('trigger_history.' + metric_name, packb([(time(), datapoint)]))
        return True

    trigger_history = unpackb(raw_trigger_history)

    # Are we (probably) triggering on the same data?
    if (new_trigger[1] == trigger_history[-1][1] and
            new_trigger[0] - trigger_history[-1][0] <= 300):
                return False

    # Update the history
    trigger_history.append(new_trigger)
    redis_conn.set('trigger_history.' + metric_name, packb(trigger_history))

    # Should we surface the anomaly?
    trigger_times = [x[0] for x in trigger_history]
    intervals = [
        trigger_times[i + 1] - trigger_times[i]
        for i, v in enumerate(trigger_times)
        if (i + 1) < len(trigger_times)
    ]

    series = pandas.Series(intervals)
    mean = series.mean()
    stdDev = series.std()

    return abs(intervals[-1] - mean) > 3 * stdDev


def run_selected_algorithm(timeseries, metric_name):
    """
    Filter timeseries and run selected algorithm.
    """
    # Get rid of short series
    if len(timeseries) < MIN_TOLERABLE_LENGTH:
        raise TooShort()

    # Get rid of stale series
    if time() - timeseries[-1][0] > STALE_PERIOD:
        raise Stale()

    # Get rid of boring series
    if len(set(item[1] for item in timeseries[-MAX_TOLERABLE_BOREDOM:])) == BOREDOM_SET_SIZE:
        raise Boring()

    try:
        ensemble = [globals()[algorithm](timeseries) for algorithm in ALGORITHMS]
        threshold = len(ensemble) - CONSENSUS
        if ensemble.count(False) <= threshold:
            if ENABLE_SECOND_ORDER:
                count = int(redis_conn.get('holtfalse'))
                count = count + 1
                redis_conn.set('holtfalse', count)
                if is_anomalously_anomalous(metric_name, ensemble, timeseries[-1][1]):
                    return True, ensemble, timeseries[-1][1]
                if verify_holt_winters(metric_name) >= 4:
                    return True, ensemble, timeseries[-1][1]
            else:
                return True, ensemble, timeseries[-1][1]

        return False, ensemble, timeseries[-1][1]
    except:
        logging.error("Algorithm error: " + traceback.format_exc())
        return False, [], 1


def get_holt_from_cabinet(full_holt_series, db):

    # Load what we know
    known_metrics = {}
    for value in full_holt_series:
        known_metrics[str(value[0])] = 1

    two_weeks = time() - 1209600
    rec_count = 0
    # traverse records
    cur = db.cursor()
    cur.jump()
    while True:
        # This is the oldest record in our cabinet
        rec = cur.get(True)
        if not rec: break
        if rec_count == 0:
            # If our first time through see if the oldest record
            # in the cabinet is located in our redis record
            # If we have seen it we can fill in the blanks from
            # our last day of redis data
            if known_metrics.has_key(str(rec[0])):
                break
            # Fill in our list with only the contents of the cabinet
            else:
                full_holt_series = []
            rec_count += 1
        if float(rec[0]) > two_weeks:
            full_holt_series.append((float(rec[0]), float(rec[1])))

    cur.disable()
    return full_holt_series

def verify_holt_winters(metric_name):
    HOLT_CACHE_DURATION = 1800
    HOLT_WINTERS_COUNT = 4
    CABINET = "/opt/skyline/src/cabinet"
    full_holt_series = []
    known_metrics = {}
    recent_holt_time = time() - HOLT_CACHE_DURATION

    db = DB()

    if not db.open(CABINET + "/" + metric_name + ".kct", DB.OREADER | DB.ONOLOCK):
        return HOLT_WINTERS_COUNT

    seen_holt = redis_conn.get('holt_' + metric_name)

    # We've put a holt_ record in redis for this metric
    if seen_holt is not None:
        full_holt_series = unpackb(seen_holt)
        # The last item in the series was seen > HOLT_CACHE_DURATION ago
        if full_holt_series[-1][0] < recent_holt_time:
            full_holt_series = get_holt_from_cabinet(full_holt_series, db)
    else:
        full_holt_series = get_holt_from_cabinet(full_holt_series, db)

    for value in full_holt_series:
        known_metrics[str(value[0])] = 1

    db.close()

    # Add the last FULL_DURATION to the cabinet data for any missing items
    raw_metric = redis_conn.mget(metric_name)
    for i, local_metric in enumerate(raw_metric):
        unpacker = Unpacker(use_list = False)
        unpacker.feed(local_metric)
        potential_new = list(unpacker)
        for value in potential_new:
            if not known_metrics.has_key(str(value[0])):
                full_holt_series.append((float(value[0]), float(value[1])))

    redis_conn.set('holt_' + metric_name, packb(full_holt_series))

    count = holtWintersDeviants(full_holt_series)
    return count


#
# Copied from https://gist.github.com/andrequeiroz
#
# This assumes a regular series input with few gaps
#
def holtWintersDeviants(full_holt_series):
    #info = additive(x, 2880, 1, 0.1, 0.0035, 0.1)
    # (24 * 60 * 60) (1 season) / 30 (step length)
    # or, we measure at 30 second intervals over a day
    # TODO: Dynamically determine the interval from the series
    m = 2880
    alpha = 0.1
    beta  = 0.0035
    gamma = 0.1
    Y = []
 
    Y = [full_holt_series[i][1] for i,values in enumerate(full_holt_series)]
 
    previous_a = sum(Y[0:m]) / float(m)
    previous_b = (sum(Y[m:2 * m]) - sum(Y[0:m])) / m ** 2
    s = [Y[i] - previous_a for i in range(m)]
    d = [0 for i in range(m)]
    y = [previous_a + previous_b + s[0]]
 
    rmse = 0

    # Find how long our number is, try to get about 1% as a min deviation
    if previous_a > 100:
        place_compare = previous_a / 300
    else:
        places = len(str(int(previous_a)))
        places -= 1 
        place_compare = (float(10 ** places) / 10)
     
    for i in range(len(Y)):
        try: 
            current_a = (alpha * (Y[i] - s[i]) + (1 - alpha) * (previous_a + previous_b))
            current_b = (beta * (current_a - previous_a) + (1 - beta) * previous_b)
            s.append(gamma * (Y[i] - previous_a - previous_b) + (1 - gamma) * s[i])
            y.append(current_a + current_b + s[i + 1])
            d.append(gamma * math.fabs(Y[i] - y[i]) + (1 - gamma) * d[i])
            previous_a = current_a
            previous_b = current_b
 
            if (d[-1] < place_compare):
                d[-1] =  place_compare
        except: 
            break
 
    deviant_count = 0
    for i in range(len(Y) - 30, len(Y) - 1):
        info = full_holt_series[i]
        hi = y[i] + 3 * d[i+m]
        lo = y[i] - 3 * d[i+m]
        if ((hi > Y[i]) and ((Y[i] + .1) > lo)):
            continue
        deviant_count += 1
 
    return deviant_count
