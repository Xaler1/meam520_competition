import os
import signal
import time
import sys

def cleanup_process(main_pid, children: list):
    while True:
        try:
            os.kill(main_pid, 0)
        except OSError:
            break
        else:
            time.sleep(1)

    for child in children:
        try:
            os.kill(child, signal.SIGKILL)
        except ProcessLookupError:
            pass

if __name__ == "__main__":
    main_pid = int(sys.argv[1])
    children = []
    for i in range(2, len(sys.argv)):
        children.append(int(sys.argv[i]))
    cleanup_process(main_pid, children)