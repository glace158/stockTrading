import redis
# redis-server
r = redis.Redis(host='localhost', port=6379, db=0) # 로컬에 띄운 Redis 서버에 연결

# 연결 테스트
r.ping()

r.set('foo', 'bar')  # key: foo value: bar
print(r.get('foo'))  # foo 라는 키의 value를 가져옴

# 리스트에 요소 추가
r.lpush("my_list", 1) 
r.rpush("my_list", 100)
 
print(r.lrange("my_list", 0, -1)) # 처음부터 끝까지 조회

# set
r.sadd("my_set", "b", "a") # "a"를 두번 추가함
# 세트의 요소 조회
print(r.smembers('myset')) 

data = {
    'name': 'John',
    "surname": 'Smith',
    "company": 'Redis',
    "age": 29
}
r.hset('user-session:123', mapping=data)

print(r.hgetall('user-session:123'))

# 트랜잭션 (명령어 한번에 실행)
with r.pipeline() as pipe:
    pipe.set('foo', 'bar')
    pipe.set('hello', 'world')
    pipe.execute()


r.delete("my_list")

# 1. 쓰기 연산 (Write)
# 키 'username'에 값을 'john_doe'로 설정
r.set('username', 'john_doe')
print(r.get('username').decode('utf-8'))  # 출력: john_doe

# 2. 수정 연산 (Update)
# 키 'username'의 값을 'john_updated'로 수정
r.set('username', 'john_updated')
print(r.get('username').decode('utf-8'))  # 출력: john_updated

# 3. 삭제 연산 (Delete)
# 키 'username' 삭제
r.delete('username')
print(r.get('username'))  # 출력: None

# 4. 전체 삭제 (Delete All)
# 데이터베이스 내의 모든 키-값 쌍을 삭제
#r.flushdb()
# 혹은
# r.flushall()  # 모든 데이터베이스의 모든 키-값 쌍을 삭제

# 5. 키 존재 확인
# 키 'username'이 존재하는지 확인
exists = r.exists('username')
print(exists)  # 출력: 1 (존재), 0 (존재하지 않음)

# 6. 키의 만료 시간 설정
# 'temp_key'에 값을 'temporary_data'로 설정하고, 10초 후에 만료되도록 설정
r.setex('temp_key', 10, 'temporary_data')

# 7. 키의 남은 시간 확인
# 'temp_key'의 남은 시간 확인
ttl = r.ttl('temp_key')
print(f'Remaining time for temp_key: {ttl} seconds')

# 8. 키의 만료 시간 제거
# 'temp_key'의 만료 시간 제거
r.persist('temp_key')

exit()