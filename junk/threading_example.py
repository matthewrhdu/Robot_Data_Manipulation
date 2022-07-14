from threading import Thread
import time


class Node:
    def __init__(self):
        self.n = 0

    def increment(self, increment: int):
        while True:
            self.n += increment
            time.sleep(0.1)  # To slow things down


if __name__ == "__main__":
    node = Node()
    t = Thread(target=node.increment, args=(5,))
    t.start()
    while node.n <= 100:
        print(node.n)
