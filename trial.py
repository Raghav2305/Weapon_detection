import cv2
import numpy as np
import imutils
import random
import string
import os

vid = r"C:\Users\ASUS\Desktop\Cloudstrats\Weapon Detection\videos\combine.mp4"
classes = []
with open('Models/obj.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

net = cv2.dnn.readNet('Models/yolov3-tiny_best.weights',
                      'Models/yolov3-tiny.cfg')

cap = cv2.VideoCapture(vid)

# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))
# size = (frame_width, frame_height)

# fourcc = cv2.VideoWriter_fourcc(*'avc1')
# # file,codec,frames per second,size
# rand_name = ''.join(random.choices(
#         string.ascii_lowercase +
#         string.ascii_uppercase,
#         k=random.choice(range(3, 9))
#     )) + '.mp4'
# print(f"\nRANDOM FILE NAME : {rand_name}\n")
# vid=os.path.join(os.getcwd(),'static','instance_output',rand_name)
# print("instance vid url : ",vid)
# result = cv2.VideoWriter(vid,fourcc, 20.0, size)

#result = cv2.VideoWriter('detect.mp4',fourcc, 20.0, size)

fr = 0
while True:
    ret, image = cap.read()
    fr += 1

    if ret:
        if fr % 5 == 0:
            # image = cv2.resize(image, (900,700))
            net.setInput(cv2.dnn.blobFromImage(image, 0.00392,
                         (416, 416), (0, 0, 0), True, crop=False))
            layer_names = net.getLayerNames()
            output_layers = [layer_names[i - 1]
                             for i in net.getUnconnectedOutLayers()]
            outs = net.forward(output_layers)

            class_ids = []
            confidences = []
            boxes = []
            Width = image.shape[1]
            Height = image.shape[0]
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.1:
                        center_x = int(detection[0] * Width)
                        center_y = int(detection[1] * Height)
                        w = int(detection[2] * Width)
                        h = int(detection[3] * Height)
                        x = center_x - w / 2
                        y = center_y - h / 2
                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        boxes.append([x, y, w, h])

            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.1)

            for i in range(len(boxes)):
                if i in indices:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    cv2.rectangle(image, (round(x), round(y)),
                                  (round(x+w), round(y+h)), (0, 0, 255), 2)
                    cv2.putText(image, label, (round(x)-10, round(y)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    print("labels:", label)

            cv2.imshow("frame", image)
            # result.write(image)
        else:
            continue
    else:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
# finally, close the window
cv2.destroyAllWindows()
