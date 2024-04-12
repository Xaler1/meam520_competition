import numpy as np

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
                return None, None, True
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


        return disp, axis, False

