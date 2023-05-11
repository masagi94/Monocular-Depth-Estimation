import cv2
import numpy as np
import os
import torch
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--video', type=str, required=True)
parser.add_argument('--dataset_name', type=str, required=True)
parser.add_argument('--midas_model', type=str, required=False, default="DPT_Large")

args = parser.parse_args()

VIDEO_PATH = args.video
DATASET_PATH = args.dataset_name
MODEL_TYPE = args.midas_model


print("\n\n******************************************************")
print("CHOSEN OPTIONS\n")
print('Video Location:', VIDEO_PATH)
print('Data Save Location:', DATASET_PATH)
print('MiDaS Model:', MODEL_TYPE)
print("******************************************************\n\n\n")

# Load a MiDas model for depth estimation
# model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
# model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
# model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", MODEL_TYPE)

# Move model to GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# Load transforms to resize and normalize the image
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if MODEL_TYPE == "DPT_Large" or MODEL_TYPE == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform


# ## Create Custom Dataset
# This method will read in a video and extract frames from it. 
# For every frame, a depth map is estimated using the state-of-the-art MiDaS model, and the resulting frames and their depth maps are saved locally to a folder specified by the user

cap = cv2.VideoCapture(VIDEO_PATH)

if not os.path.exists(DATASET_PATH):
    os.mkdir(DATASET_PATH)

count = 0
while(cap.isOpened()):
  # Capture frame-by-frame
    ret, img = cap.read()

    if not ret:
        break
    img = img[:,:,::-1]
    # Apply input transforms
    input_batch = transform(img).to(device)

    # Prediction and resize to original resolution
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    depth_map = (depth_map*255).astype(np.uint8)
    depth_map = cv2.applyColorMap(depth_map , cv2.COLORMAP_BONE)
    
    cv2.imwrite(DATASET_PATH+'/' + str(count) + '.png', img)
    np.save(DATASET_PATH+'/' + str(count) + '.npy', depth_map[:,:,0])
    count += 1
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break


cv2.destroyAllWindows()
cap.release()