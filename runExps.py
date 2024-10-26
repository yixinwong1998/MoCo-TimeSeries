import multiprocessing
import time
import random
import os


def run_script(args):
    # construct the command line string with the arguments
    # please modify the path to main.py and augments according to your project
    para_go = f"python main.py --feature_dim {args[0]} \
    --m {args[1]} \
    --temperature {args[2]} \
    --momentum {args[3]} \
    --k {args[4]} \
    --batch_size {args[5]} \
    --epochs {args[6]} "

    time.sleep(random.random())  # avoid the processors crash
    start = time.time()
    # execute the command using os.system
    os.system(para_go)
    end = time.time()
    print(f"Running with args: {args}, execution time: {(end - start) / 60} mins")

    # with open("./trainingRecord.txt", "a") as f:
    #     f.write(f"episode {args[0]}, lr {args[1]}, ppoe {args[2]}, wRa {args[3]}, wRb {args[4]}, wRc {args[5]}, wRd {args[6]}, pR {args[7]}, EXP{exp}. \n")
    #     f.write(f"Execution time: {(end - start) / 60} mins. \n")


if __name__ == "__main__":
    # create a list of arguments to pass to main.py
    inputs = [
        [128, 2048, 0.07, 0.999, 10, 64, 4,],
        [128, 4096, 0.07, 0.999, 10, 64, 4,],
        [128, 2048, 0.07, 0.999, 10, 64, 6,],
        [128, 4096, 0.07, 0.999, 10, 64, 6,],
    ]

    # create a pool of workers
    pool = multiprocessing.Pool(4)
    # map the run_script function to the inputs
    pool.map(run_script, inputs)
