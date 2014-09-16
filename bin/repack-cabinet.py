#!/usr/local/bin/python

from redis import StrictRedis, WatchError
from msgpack import packb, unpackb, Unpacker
from kyotocabinet import *
import sys
from time import time
import os

# create the database object
db = DB()
db2 = DB()
# open the database

redis_conn = StrictRedis(unix_socket_path = '/tmp/redis.sock')

full_list = list(redis_conn.smembers('system.unique_metrics'))
if len(full_list) == 0:
  print "No metrics"
  exit()

count = 0
start = time()
for metric in full_list:
    print "Pulled %s from the list" % (metric)
    CABINET = "/opt/skyline/src/cabinet/"
    count += 1
    if not db.open(CABINET + metric + ".kct#dfunit=8", DB.OREADER | DB.ONOLOCK):
      print >>sys.stderr, metric + "open error: " + str(db.error())
      next

    if not db2.open(CABINET + "../cabinet-tmp/" + metric + ".kct#dfunit=8", DB.OWRITER | DB.OCREATE):
      print >>sys.stderr, metric + "open error: " + str(db.error())

    two_weeks = time() - 1209600
    rec_count = 0
    # traverse records
    cur = db.cursor()
    cur.jump()
    while True:
        # This is the oldest record in our cabinet
        rec = cur.get(True)
        if not rec: break
        try:
           if float(rec[0]) > two_weeks:
              db2.set(rec[0], rec[1])
        except ValueError,e:
           print "Error ",e," on ", rec[0]

    cur.disable()
    db.close()
    db2.close()
    if (count % 100) == 0:
       print "%s keys.  Rate: %s" % (count, (100/(time() - start)))
       os.system("mv /opt/skyline/src/cabinet-tmp/* /opt/skyline/src/cabinet")
       start = time()
       print "%s keys.  Rate: %s" % (count, (100/(time() - start)))

os.system("mv /opt/skyline/src/cabinet-tmp/* /opt/skyline/src/cabinet")
start = time()
print "%s keys.  Rate: %s" % (count, (100/(time() - start)))
