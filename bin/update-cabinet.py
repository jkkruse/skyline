#!/usr/local/bin/python

from redis import StrictRedis, WatchError
from msgpack import packb, unpackb, Unpacker
from kyotocabinet import *
import sys
import time

# create the database object
db = DB()
# open the database

redis_conn = StrictRedis(unix_socket_path = '/tmp/redis.sock')

full_list = list(redis_conn.smembers('system.unique_metrics'))
if len(full_list) == 0:
  print "No metrics"
  exit()

count = 0
start = time.time()
for metric in full_list:
   count += 1
   if not db.open("/opt/skyline/src/cabinet/" + metric + ".kct", DB.OWRITER | DB.OCREATE):
     print >>sys.stderr, metric + "open error: " + str(db.error())

   raw_metric = redis_conn.mget(metric)
   for i, metric_name in enumerate(raw_metric):
      unpacker = Unpacker(use_list = False)
      unpacker.feed(metric_name)
      timeseries = list(unpacker)
      for value in timeseries:
         if db.check(value[0]) < 0:
            db.set(value[0], value[1])
         #db.set(value[0], value[1])
   db.close()
   if (count % 100) == 0:
      print "%s keys.  Rate: %s" % (count, (100/(time.time() - start)))
      start = time.time()
