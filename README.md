# Monocular-Depth-Estimation

CS 682 Final Project

Depth Estimation from single Monocular Camera

This project contains code for depth estimation from a single monocular camera, using a deep learning approach. The codebase includes a pre-trained model and instructions for training on custom datasets.

There are two developers of the project at hand, namely Mauricio and Wasef.

This project was developed as a group project implementation as a part of the requirements of the course CS 682 at George Mason University under the supervision and guidance of Professor Jana Kosecka.

******************************************************************************
To create the Conda environment needed to run our code, run the following:


	conda create --name tf-gpu python=3.9
	conda activate tf-gpu
	conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
	python -m pip install "tensorflow<2.11"
	conda install ipykernel
	ipython kernel install --user --name=tf-gpu
	conda install pandas
	pip install opencv-contrib-python
	conda install -c conda-forge matplotlib
	conda update Pillow

Then, install the cuda toolkit:

	https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10


We did not include our dataset in this submission because the total size was 21GB.
If you'd like to use your own dataset, record a video and use the Create_Dataset.py file to create depth maps for it.
In order for Create_Dataset.py to work, you will need to configure a new environment like so:

	conda env create -f environment.yaml
	conda activate midas-py310 

We followed the isl-team's github for midas setup: https://github.com/isl-org/MiDaS/tree/master
We included the .yaml file with our submission.

******************************************************************************
An example run of our project that creates a dataset from a video, trains a model on it, uses a webcam to live-view the model estimations, and generates point clouds from it:

With midas-py310 environment:

		python Create_Dataset.py --video=Data/Train_Vid.mp4 --dataset_name=train_set

With tf-gpu environment:

		python Train_New_Model.py --train_dataset=Data\Train_Set --test_dataset=Data\Test_Set --epochs=15 --learning_rate=.001
		python Webcam_Model_Estimation.py --model=trained_model
		python Generate_Point_Clouds.py --dataset_path=Data/Test_Set --model_path=trained_model
