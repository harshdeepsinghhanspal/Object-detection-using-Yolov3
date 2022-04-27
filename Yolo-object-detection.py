import cv2
import numpy as np

cap=cv2.VideoCapture(0)
whT=320 #Weight and Height of the Target
confThreshold=0.5

#colours for box and texts
b=0
g=0
r=255


classesFile='coco.names'
classes=[]
with open(classesFile,'rt') as f:
    classNames=f.read().rstrip('\n').split('\n')
#print(classNames)
#print(len(classNames))


#Slower frame rates but higher confidence
modelConfiguration='yolov3.cfg'
modelWeights='yolov3.weights'

#Faster frame rates but lower confidence
#modelConfiguration='yolov3-tiny.cfg'
#modelWeights='yolov3-tiny.weights'

net=cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObjects(outputs,img):
    hT, wT, cT=img.shape
    bbox=[]
    classIds=[]
    confs=[]

    for output in outputs:
        for det in output:
            scores=det[5:]
            classId=np.argmax(scores)
            confidence=scores[classId]
            if confidence > confThreshold:
                w,h=int(det[2]*wT), int(det[3]*hT) #width and height from array
                x,y=int((det[0]*wT)-w/2), int((det[1]*hT)-h/2) #same as above for x,y
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
    #print(len(bbox))
    indices=cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nms_threshold=0.3)
    #print(indices)
    for i in indices:
        #i=i[0]
        box=bbox[i]
        x,y,w,h=box[0],box[1],box[2],box[3]
        cv2.rectangle(img,(x,y),(x+w,y+h),(b,g,r),2)
        cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%', (x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(b,g,r),2)

while True:
    success, img=cap.read()

    blob=cv2.dnn.blobFromImage(img,1/255,(whT,whT),[0,0,0],1,crop=False)
    net.setInput(blob)

    layerNames=net.getLayerNames()
    #print(layerNames)
    outputNames=[layerNames[i-1] for i in net.getUnconnectedOutLayers()]
    #print(outputNames)

    #print(net.getUnconnectedOutLayers()) #Only output layers

    outputs=net.forward(outputNames)
    #print(outputs[0].shape) #no of boxes=300
    #print(outputs[1].shape) #no of boxes=1200
    #print(outputs[2].shape) #no of boxes=4800
    #print(outputs[0][0])    #x,y,w,h, confidence(5th val) hence these are additional 5 to the 80 values in coco.names


    findObjects(outputs,img)
    cv2.imshow("Image",img)
    cv2.waitKey(1)