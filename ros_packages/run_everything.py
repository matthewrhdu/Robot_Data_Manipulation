from multiprocessing import Process
import os

def run_command(cmd):
    stream = os.popen(cmd)
    print(stream.read())

def main():
    mods = ['camera', 'processor', 'controller', 'robot']

    p = [Process(target=run_command, args=(f"ros2 run in_hand_control {mod}",)) for mod in mods]

    for pro in p:
        pro.start()


if __name__ == "__main__":
    main()