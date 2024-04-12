import numpy as np
from scipy.interpolate import splprep, splev, make_interp_spline
from lib.calcAngDiff import calcAngDiff

from lib.IK_position_null import IK

class Trajectory:
    def __init__(self):
        pass

    def __call__(self, T0e, t) -> tuple:
        pass


class Waypoints(Trajectory):
    def __init__(self, points):
        self.points = points
        self.num_points = len(points)
        self.current_point = 0
        print("Starting waypoints trajectory")

    def __call__(self, T0e, t):
        target = self.points[self.current_point]

        dist, ang = IK.distance_and_angle(T0e, target)
        if dist < 0.01 and ang < 0.01:
            print(f"Reached waypoint {self.current_point}")
            self.current_point += 1
            if self.current_point >= self.num_points:
                print("Finished waypoints trajectory")
                return None, None, None, None, True
            target = self.points[self.current_point]

        disp, axis = IK.displacement_and_axis(target, T0e)
        if np.linalg.norm(disp) > 0.02:
            # normalize to 0.2 m/s
            disp = 0.2 * disp / np.linalg.norm(disp)
        if np.linalg.norm(axis) > 0.02:
            # normalize to 0.5 rad/s
            axis = 0.2 * axis / np.linalg.norm(axis)

        # ease into the trajectory based on time
        if t < 0.9:
            disp *= t + 0.1
            axis *= t + 0.1


        return disp, axis, None, None, False

class Spline(Trajectory):
    def __init__(self, points):
        self.points = np.array(points)
        locs = self.points[:, :3, 3]
        # insert points between first and second until at least 4 points:
        while len(locs) < 4:
            locs = np.insert(locs, 1, (locs[0] + locs[1]) / 2, axis=0)
        x = locs[:, 0]
        y = locs[:, 1]
        z = locs[:, 2]
        self.progress_spline, u = splprep([x, y, z], s=0.1)
        self.easing_time = 0.5
        total_length = 0
        for i in range(1, len(locs)):
            total_length += np.linalg.norm(locs[i] - locs[i - 1])
        self.total_time = total_length / 0.07
        ease_start = self.easing_time / self.total_time
        ease_end = 1 - ease_start
        x = [0.0, ease_start, 0.5, ease_end, 1.0]
        y = [0.0, 0.1, 0.5, 0.9, 1.0]
        self.ease_spline, u = splprep([x, y], s=0, k=3)

        print("Starting spline trajectory")
        print(f"Estimated time: {self.total_time} seconds")

    def __call__(self, T0e, t):
        progress = splev(t / self.total_time, self.ease_spline)[1]
        speed_multiplier = splev(t / self.total_time, self.ease_spline, der=1)[1]

        if progress >= 1.0:
            print("Finished spline trajectory")
            return None, None, None, None, True
        x, y, z = splev(progress, self.progress_spline)
        dx, dy, dz = splev(progress, self.progress_spline, der=1)

        axis = calcAngDiff(self.points[-1][:3, :3], T0e[:3, :3])
        if np.linalg.norm(axis) > 0.02:
            # normalize to 0.5 rad/s
            axis = 0.5 * axis / np.linalg.norm(axis)

        xdes = np.array([x, y, z])
        vdes = np.array([dx, dy, dz]) * speed_multiplier
        return vdes, axis, xdes, None, False



