import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


class ObjectTracker:
    count = 0

    def __init__(self, box, class_name, distance):
        ObjectTracker.count += 1
        self.id         = ObjectTracker.count
        self.class_name = class_name
        self.distance   = distance
        self.hits       = 1
        self.no_match   = 0
        self.history    = []
        self.kf         = KalmanFilter(dim_x=8, dim_z=4)
        self.kf.F       = np.eye(8)
        for i in range(4):
            self.kf.F[i, i+4] = 1
        self.kf.H       = np.zeros((4, 8))
        self.kf.H[:4, :4] = np.eye(4)
        self.kf.R      *= 10
        self.kf.P      *= 100
        self.kf.Q      *= 0.1
        self.kf.x[:4]   = self._to_state(box)

    def _to_state(self, box):
        x1, y1, x2, y2 = box
        return np.array([[(x1+x2)/2], [(y1+y2)/2],
                         [x2-x1],     [y2-y1]], dtype=float)

    def _to_box(self):
        cx, cy, w, h = self.kf.x[:4].flatten()
        return [cx-w/2, cy-h/2, cx+w/2, cy+h/2]

    def predict(self):
        self.kf.predict()
        return self._to_box()

    def update(self, box, distance):
        self.kf.update(self._to_state(box))
        self.distance  = distance
        self.hits     += 1
        self.no_match  = 0
        self.history.append(self._to_box())


class MultiObjectTracker:
    def __init__(self, iou_threshold=0.15, max_age=2):
        self.trackers      = []
        self.iou_threshold = iou_threshold
        self.max_age       = max_age

    def _iou(self, b1, b2):
        xi1 = max(b1[0], b2[0]); yi1 = max(b1[1], b2[1])
        xi2 = min(b1[2], b2[2]); yi2 = min(b1[3], b2[3])
        inter = max(0, xi2-xi1) * max(0, yi2-yi1)
        a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        return inter / (a1 + a2 - inter + 1e-6)

    def update(self, detections):
        predicted   = [t.predict() for t in self.trackers]
        matched_d   = set()
        matched_t   = set()

        if self.trackers and detections:
            cost = np.zeros((len(detections), len(self.trackers)))
            for d, det in enumerate(detections):
                for t, pred in enumerate(predicted):
                    cost[d, t] = 1 - self._iou(det['box'], pred)
            di, ti = linear_sum_assignment(cost)
            for d, t in zip(di, ti):
                if cost[d, t] < (1 - self.iou_threshold):
                    self.trackers[t].update(
                        detections[d]['box'], detections[d]['distance']
                    )
                    matched_d.add(d)
                    matched_t.add(t)

        for d, det in enumerate(detections):
            if d not in matched_d:
                self.trackers.append(
                    ObjectTracker(det['box'], det['class_name'], det['distance'])
                )

        for i, t in enumerate(self.trackers):
            if i not in matched_t:
                t.no_match += 1
        self.trackers = [t for t in self.trackers if t.no_match <= self.max_age]
        return self.trackers
