import cv2
import math
import time
import torch
import imutils
import numpy as np
from scipy.spatial import distance as dist

from modules.config import camera_no
from modules.detection import detect_people

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# set CUDA as the preferable 
# backend and target if GPU is available
if torch.cuda.is_available():
    print("")
    print("[INFO] Looking for GPU")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


labelsPath = "coco-names/coco_names.txt"
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)

COLORS = np.random.randint(0, 
                        255, 
                        size=(len(LABELS), 3),
                        dtype="uint8")

weightsPath = "models/yolov3.weights"
configPath = "models/yolov3.cfg"
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# load our serialized face detector model from disk
print("loading face detector model...")
# confidence threshold for face mask classification
confidence_threshold = 0.5
prototxtPath = "models/deploy.prototxt"
weightsPath = "models/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model
model_store_dir= "models/model (improved)/model"
maskNet = load_model(model_store_dir)

cap = cv2.VideoCapture(camera_no)       #Start Video Streaming

while (cap.isOpened()):
    ret, image = cap.read()

    if ret == False:
        break

    image = cv2.resize(image, (760, 640))
    # cv2.namedWindow("output", cv2.WINDOW_NORMAL)

    # Display text "PRESS 'q' TO EXIT"
    cv2.putText(image, "PRESS 'q' TO EXIT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3) 

    (H, W) = image.shape[:2]
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    blob = cv2.dnn.blobFromImage(image, 
                                1/255.0, 
                                (416, 416), 
                                swapRB=True, 
                                crop=False)

    results = detect_people(image, net, ln,
                            personIdx=LABELS.index("person"))
    # print("Detect p:  ", results)    

    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    print("Time taken to predict the image: {:.6f}seconds".format(end-start))
    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.1 and classID == 0:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    ind = []
    for i in range(0, len(classIDs)):
        if (classIDs[i] == 0):
            ind.append(i)

    a = []
    b = []
    #colour = (0, 255, 0)
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            a.append(x)
            b.append(y)
            #cv2.rectangle(image, (x, y), (x + w, y + h), colour, 2)

    distance = []
    nsd = []
    for i in range(0, len(a) - 1):
        for k in range(1, len(a)):
            if (k == i):
                break
            else:
                x_dist = (a[k] - a[i])
                y_dist = (b[k] - b[i])
                d = math.sqrt(x_dist * x_dist + y_dist * y_dist)
                distance.append(d)
                if (d <= 100.0):
                    nsd.append(i)
                    nsd.append(k)
                nsd = list(dict.fromkeys(nsd))

    colour = (0, 0, 255)
    for i in nsd:
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        cv2.rectangle(image, (x, y), (x + w, y + h), colour, 1)
        text = "Alert"
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1)

    colour = (138, 68, 38)
    if len(idxs) > 0:
        for i in idxs.flatten():
            if (i in nsd):
                break
            else:
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                cv2.rectangle(image, (x, y), (x + w, y + h), colour, 1)
                text = "SAFE"
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1)


    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (416, 416), (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # _____________________________________________________________________ #
            #  Counting Sources

            serious = set()
            abnormal = set()

            # ensure there are *at least* two people detections (required in
            # order to compute our pairwise distance maps)
            if len(results) >= 2:
                # extract all centroids from the results and compute the
                # Euclidean distances between all pairs of the centroids
                centroids = np.array([r[2] for r in results])
                D = dist.cdist(centroids, centroids, metric="euclidean")

                # loop over the upper triangular of the distance matrix
                for i in range(0, D.shape[0]):
                    for j in range(i + 1, D.shape[1]):
                        # check to see if the distance between any two
                        # centroid pairs is less than the configured number of pixels
                        if D[i, j] < 50:
                            # update our violation set with the indexes of the centroid pairs
                            serious.add(i)
                            serious.add(j)
                        # update our abnormal set if the centroid distance is below max distance limit
                        if (D[i, j] < 80) and not serious:
                            abnormal.add(i)
                            abnormal.add(j)

            # loop over the results
            for (i, (prob, bbox, centroid)) in enumerate(results):
                # extract the bounding box and centroid coordinates, then
                # initialize the colour of the annotation
                (startX, startY, endX, endY) = bbox
                (cX, cY) = centroid
                colour = (0, 255, 0)

                # if the index pair exists within the violation/abnormal sets, then update the colour
                if i in serious:
                    colour = (0, 0, 255)
                elif i in abnormal:
                    colour = (0, 255, 255) #orange = (0, 165, 255)

                # draw (1) a bounding box around the person and (2) the
                # centroid coordinates of the person,
                cv2.rectangle(image, (startX, startY), (endX, endY), colour, 2)
                cv2.circle(image, (cX, cY), 2, colour, 2)


            # _____________________________________________________________________#

            mask_prob = maskNet.predict(face, batch_size=100)
            mask_pred = np.argmax(mask_prob, axis=1)[0]
            mask_prob = round(np.max(mask_prob), 4) * 100
            
            if mask_pred == 0:
                label = 'Mask'
                colour = (0, 255, 0)    # green
            elif mask_pred == 1:
                label = 'Mask (Incorrect)'
                colour = (0, 150, 200)
            elif mask_pred == 2:
                label = 'No Mask'
                colour = (0, 0, 255)    # red

            label = "{}: {:.2f}%".format(label, mask_prob)
            c = cv2.putText(image, label, (startX + 400, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1)
            x = cv2.rectangle(image, (startX, startY), (endX, endY), colour, 1)
            
            text = "Total serious violations: {}".format(len(serious))
            # cv2.putText(image, text, 
            #             (10, image.shape[0] - 55),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.70, 
            #             (0, 0, 255), 2)

            text1 = "Total abnormal violations: {}".format(len(abnormal))
            # cv2.putText(image, text1, 
            #             (10, image.shape[0] - 25),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.70, 
            #             (0, 255, 255), 2)


            print("End of classifier")

    # imS = cv2.resize(image, (960, 540))
    # ver = np.vconcat([image, ig])
    cv2.imshow("Image", image)
    # cv2.imshow("Face", x)
    # cv2.imshow("Face", label)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()