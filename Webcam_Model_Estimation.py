import argparse
import numpy as np 
import os
import time
import tensorflow as tf
import cv2
parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, required=True)
parser.add_argument('--image_size', type=int, required=False, default=256)

args = parser.parse_args()


PATH = args.model
SIZE = (args.image_size,args.image_size)


print('model location:', PATH)
print('Input image size:', SIZE)


if(not os.path.exists(PATH)):
	print("ERROR: '" +PATH + "' DOES NOT EXIST")
	exit()

imported_model = tf.keras.models.load_model("trained_model")
print("model imported...")

cap = cv2.VideoCapture(0)
count = 0
print("reading frames...")

while True:

    success, img = cap.read()

    start = time.time()

    image_ = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print(image_)
    image_ = cv2.resize(image_, SIZE)
    image_ = tf.image.convert_image_dtype(image_, tf.float32)

    try:
    	pred = imported_model.predict(image_[None,:,:,:])      
    except:
    	print("Predicting on images failed. Most likely reason is the input image size is not what the model was trained on.\n" +
    		  "Set the --size field to the image size the model was trained on (256 or 512)\n")
    	break

    pred = cv2.resize(pred.squeeze(), SIZE)
    pred = (pred*256).astype(np.uint8)
    pred = cv2.applyColorMap(pred, cv2.COLORMAP_JET)
    end = time.time()
    totalTime = end - start
    fps = 1 / totalTime
   
    cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
    cv2.imshow("Input",(img.squeeze()))

    cv2.imshow("Depth Map",(pred))

    count += 1
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()