import redis

r = redis.Redis(host='localhost', port=6379, db=0)

print(r.hgetall('user-session:123'))
r.flushall()
exit()