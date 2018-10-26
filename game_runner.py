import sys
import subprocess
from threading import Thread
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import time

CMD = "./bin/runner"

def run_game(args):
    uid, parameters = args

    str_parameters = str(len(parameters)) + " " + " ".join(map(str, parameters))

    attempts = 0

    while attempts < 5:
        attempts += 1
        try:
            proc = subprocess.run(CMD,
                                  input = str_parameters.encode('ascii'),
                                  stdout = subprocess.PIPE,
                                  stderr = subprocess.PIPE)
            output = proc.stdout.decode('ascii').rstrip()
            result = output.split(" ")
            return result
        except subprocess.TimeoutExpired:
            proc.kill()
            time.sleep(1)


def run_games(games):
    executor = ThreadPoolExecutor(max_workers=4)

    return executor.map(run_game, games)

def main():
    f = open("dist_input.txt", "r")
    s = f.read()
    p = list(map(float, s.split(" ")))
    p = p[1:]

    games = [ (1, p), (2, p), (3, p), (4, p) ]
    
    for result in run_games(games):
        print(result)
    
if __name__ == "__main__":
    main()
