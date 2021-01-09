import Yolo.config as config
from Yolo.detection import detect
from Yolo.sort import Sort
from PIL import Image
import imutils
import cv2
import os
from tqdm import tqdm
import warnings
warnings.simplefilter("ignore", UserWarning)

def detect_track(video_path):
	# load the COCO class labels our YOLO model was trained on
	labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
	LABELS = open(labelsPath).read().strip().split("\n")

	# derive the paths to the YOLO weights and model configuration
	weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
	configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

	# load our YOLO object detector trained on COCO dataset (80 classes)
	print("[INFO] loading YOLO from disk...")
	#net = cv2.dnn.readNet(weightsPath, configPath)
	net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

	if config.USE_GPU:
		net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
		net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

	# determine only the *output* layer names that we need from YOLO
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	print("[INFO] accessing video stream...")
	vs = cv2.VideoCapture(video_path)
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
	total_frames = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total_frames))

	tracker = {"car": Sort(), "person": Sort()}
	coords = {"car": dict(), "person": dict()}
	predictions = dict()
	for i in range(total_frames):
		predictions['car' + str(i)] = dict()
	frameIndex = 0
	max_frames = 50 #int(vs.get(prop))
	pbar = tqdm(total=max_frames)
	while frameIndex < max_frames:
		pbar.update(1)
		(grabbed, frame) = vs.read()
		if not grabbed:
			break
		frame = imutils.resize(frame, width=700)
		results = detect(frame, net, ln)
		detections = {"car":[], "person":[]}
		# loop over the results
		for (i, (bbox, prob, label)) in enumerate(results):
			detections[LABELS[label]].append(bbox+prob)
		for k in tracker:
			if detections[k] != []:
				tracks = tracker[k].update(detections[k])
				boxes = []
				indexIDs = []
				memory = {}
				for track in tracks:
					boxes.append([track[0], track[1], track[2], track[3]])
					indexIDs.append(int(track[4]))
					memory[indexIDs[-1]] = boxes[-1]
				if len(boxes) > 0:
					coords[k]['frame-' + str(frameIndex)] = dict()
					i = int(0)
					for box in boxes:
						# extract the bounding box coordinates
						(x1, y1) = (int(box[0]), int(box[1]))
						(x2, y2) = (int(box[2]), int(box[3]))
						coords[k]['frame-' + str(frameIndex)]['car' + str(indexIDs[i])] = [x1, y1, x2, y2]
						if k == "car":
							car = frame[max(0, y1):min(y2, frame.shape[0]), max(0, x1):min(x2, frame.shape[1]), :]
							car_img = Image.fromarray(car)
							if car.shape[0] > 50 and car.shape[1] > 50:
								#predictions['car' + str(indexIDs[i])]['frame-' + str(frameIndex)] = predict(car, VMMR_model)
								car_img.save("./Yolo/cropped_cars/car#{}_frame#{}.jpg".format(indexIDs[i], frameIndex))
							else:
								del coords[k]['frame-' + str(frameIndex)]['car' + str(indexIDs[i])]
						i += 1
		frameIndex += 1

detect_track(r"D:\Nada\PFE\yolo\videos\17_17 38.mp4")