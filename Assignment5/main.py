import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

VIDEO = "input.mp4"
OUTPUT = "output.mp4"
LINE_START, LINE_END = (960, 250), (960, 850)                              # virtual line
ZONE = np.array([[1350, 570], [1900, 570], [1900, 760], [1350, 760]])      # restricted zone

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
cap.release()
writer = cv2.VideoWriter(OUTPUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

model = YOLO("yolov8m.pt")
line = sv.LineZone(start=sv.Point(*LINE_START), end=sv.Point(*LINE_END))
zone = sv.PolygonZone(polygon=ZONE)

box_ann = sv.BoxAnnotator()
line_ann = sv.LineZoneAnnotator()
zone_ann = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.RED)

alerted = set()

for result in model.track(VIDEO, stream=True, persist=True, classes=[0],
                          tracker="bytetrack.yaml", verbose=False):
    frame = result.orig_img
    det = sv.Detections.from_ultralytics(result)
    in_zone = False

    if det.tracker_id is not None:
        line.trigger(det)
        inside = zone.trigger(det)
        in_zone = inside.any()

        for tid, is_in in zip(det.tracker_id, inside):
            if is_in and tid not in alerted:
                alerted.add(tid)
                print(f"ALERT: person {tid} entered restricted zone")

    frame = box_ann.annotate(frame, det)
    frame = line_ann.annotate(frame, line)
    frame = zone_ann.annotate(frame)
    if in_zone:
        cv2.putText(frame, "ALERT: RESTRICTED ZONE", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    writer.write(frame)
    cv2.imshow("Counter", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

writer.release()
cv2.destroyAllWindows()
print(f"IN: {line.in_count}  OUT: {line.out_count}")
print(f"Video saved: {OUTPUT}")