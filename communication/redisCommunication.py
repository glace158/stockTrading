import redis
# redis-server

class RedisCom:
    def __init__(self, port = 6379):
                
        self.port = port
        self.client = redis.Redis(host='localhost', port=port, db=0) # 로컬에 띄운 Redis 서버에 연결
    
    def sendData(self, key, data):
        with self.client.pipeline() as pipe:
            pass

    def loadData(self, key):
        with self.client.pipeline() as pipe:
            pass
