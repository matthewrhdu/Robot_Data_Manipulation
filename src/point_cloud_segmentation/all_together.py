import numpy as np
from typing import Optional
import open3d as o3d
from threading import Thread
from queue import Queue
from time import sleep
from typing import List

from Algorithms_Builtins import *
import global_registration2 as gr
# import PointOfViewCamera as PoVC

NUM_VIEWS = 5


def _filter(plane_removed_img: np.ndarray, filter_spec: List) -> np.ndarray:
    mask = np.full((plane_removed_img.shape[0],), True)
    for i in range(3):
        lower_mask = plane_removed_img[:, i] > filter_spec[i][0]
        upper_mask = plane_removed_img[:, i] < filter_spec[i][1]
        first = np.logical_and(lower_mask, upper_mask)

        mask = np.logical_and(mask, first)

    filtered = []
    for m in range(len(mask)):
        if mask[m]:
            filtered.append(plane_removed_img[m])
    filtered = np.array(filtered)
    return filtered


class MergedImage:
    accumulated: Optional[np.ndarray]

    def __init__(self) -> None:
        self.accumulated = None
        self.lineup = Queue(NUM_VIEWS)
        self.domain = [[], [], []]
        self.n = 0
        self.t = 0

    def take_picture(self):
        data = np.load(f"npy/img{self.n}.npy")
        self.lineup.put(data)
        print("click!!!")
        self.n += 1

    def accumulate(self):
        img = self.lineup.get(block=True)
        plane_removed_img = run_ransac(img, 0.0075)
        if self.accumulated is None:
            self.accumulated = plane_removed_img
            self._get_domain(plane_removed_img)
        else:
            filtered = _filter(plane_removed_img, self.domain)

            stat_outliers_range = []
            for i in range(3):
                mean = np.average(filtered[:, i])
                std = np.std(filtered[:, i])
                stat_outliers_range.append([mean - std, mean + std])

            filtered = _filter(filtered, stat_outliers_range)

            print("here")
            self.accumulated = gr.main(filtered, self.accumulated, voxel_size=0.005)
            np.save(f"sub/sub{self.t}.npy", self.accumulated)
            self.t += 1

    def _get_domain(self, data):
        for i in range(3):
            self.domain[i] = [min(data[:, i]), max(data[:, i])]


def next_thread(machine: MergedImage):
    for n in range(NUM_VIEWS):
        machine.take_picture()
        sleep(1)


def main():
    machine = MergedImage()
    camera_thread = Thread(target=next_thread, args=(machine,))
    camera_thread.start()

    for _ in range(5):
        machine.accumulate()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(machine.accumulated)
    o3d.visualization.draw_geometries([pcd], width=800, height=600)


if __name__ == "__main__":
    main()
