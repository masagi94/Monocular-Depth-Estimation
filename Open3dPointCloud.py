import argparse
import numpy as np 
import os
import time
import tensorflow as tf
import cv2

import open3d as o3d


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


success, img = cap.read()
pcd = o3d.geometry.Image((img).astype(np.uint8))

vis = o3d.visualization.Visualizer()
vis.create_window()




while True:

    success, img = cap.read()

    start = time.time()

    image_ = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print(image_)
    image_ = cv2.resize(image_, SIZE)
    img = cv2.resize(img, (1024,1024))
    image_ = tf.image.convert_image_dtype(image_, tf.float32)

    try:
    	pred = imported_model.predict(image_[None,:,:,:])      
    except:
    	print("Predicting on images failed. Most likely reason is the input image size is not what the model was trained on.\n" +
    		  "Set the --size field to the image size the model was trained on (256 or 512)\n")
    	break

    pred = cv2.resize(pred.squeeze(), (1024,1024))
    pred = (pred*256).astype(np.uint8)
    pred = cv2.applyColorMap(pred, cv2.COLORMAP_JET)


    vis.remove_geometry(pcd,False)
    
    # color_raw = o3d.io.read_image(input_img)
    # depth_raw = o3d.io.read_image(input_depth)

    color_raw = o3d.geometry.Image((img).astype(np.uint8))
    depth_raw = o3d.geometry.Image((pred).astype(np.uint8))

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -20, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        
    # add pcd to visualizer
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.001)




    # end = time.time()
    # totalTime = end - start
    # fps = 1 / totalTime

    # shape = img.shape
    # cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
    # cv2.imshow("Input",(img.squeeze()))

    # pred = cv2.resize(pred, (shape[0], shape[1]))
    # cv2.imshow("Depth Map",(pred))

    # count += 1
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()