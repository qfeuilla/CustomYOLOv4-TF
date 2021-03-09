from model import head
from model import body

def YOLOv4(input_data, n_classes=10):
    out = body.YOLOv4_body(input_data)
    out = head.YOLOv4_head(out, n_classes)
    return out

