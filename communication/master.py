import subprocess
import os

def run_pythonfile(runcmd, fileName):
    subprocess.run([runcmd, str(os.path.dirname(__file__)) + '/' + fileName])


if __name__ == "__main__":
    os.popen("find ./ -name ._\* -delete")
    # 32비트 Python 실행
    run_pythonfile('python3', 'redisTXTest.py')
    
    # 64비트 Python 실행
    run_pythonfile('python3', 'redisRXTest.py')