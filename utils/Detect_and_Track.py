from utils.detection import detect
from utils.sort import Sort
from PIL import Image
import cv2, imutils, commentjson, torch
import numpy as np
from tqdm import tqdm
import torchvision.transforms.functional as TF
from torchvision import transforms, models
import torch.nn as nn
import warnings, os, glob, shutil
warnings.simplefilter("ignore", UserWarning)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("[INFO] System is running on {}:".format(device))
class_names = []
with open(r"./data/class_names_2522c.txt", 'r') as f:
	for line in f:
		class_names.append(line.rstrip('\n'))

def prepare_model(config):
	if config["model"] == "resnet18":
		model = models.resnet18(pretrained=True)
	elif config["model"] == "resnet50":
		model = models.resnet50(pretrained=True)
	elif config["model"] == "resnet101":
		model = models.resnet101(pretrained=True)
	else:
		print("Unrecognized model. Please pick resnet18, resnet50 or resnet101 in the configuration file")

	num_ftrs = model.fc.in_features
	model.fc = nn.Linear(num_ftrs, len(class_names))
	if config["selected_model_path"].split(".")[-1] == "pth":
		model.load_state_dict(torch.load(config["selected_model_path"])).to(device)
	elif config["selected_model_path"].split(".")[-1] == "t7":
		model = torch.load(config["selected_model_path"])
	else:
		print("Unsuitable file extension.")
	return model

def predict(image, model, config, k):
	data_transforms = transforms.Compose([
		transforms.Resize(config["input_shape"]),
		transforms.CenterCrop(config["input_shape"]),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

	img = data_transforms(transforms.ToPILImage()(TF.to_tensor(image))).reshape(
		(1, 3, config["input_shape"], config["input_shape"]))
	img = img.to(device).float()
	output = model(img)
	return torch.topk(output.data, k)

def merge_predictions(predictions):
	# Prediction probabilities fusion
	predictions = {k: v for k, v in predictions.items() if v}
	pred_fus = dict()
	for car in predictions:
		pred_fus[car] = dict()
		for c in range(len(class_names)):
			pred_fus[car][c] = 0
		for fr in predictions[car]:
			for i in range(len(predictions[car][fr][0])):
				for j in range(3):
					pred_fus[car][predictions[car][fr][1][i][j].cpu().numpy().item()] += predictions[car][fr][0][i][j].cpu().numpy().item()
	# Get the final prediction
	final_pred = dict()
	for car in predictions:
		final_pred[car] = []
		final_pred[car].append(sorted(pred_fus[car], key=pred_fus[car].get, reverse=True)[:3])
	return final_pred

def detect_track(video_path):
	with open(r"./config.json") as json_file:
		config = commentjson.load(json_file)

	if not os.path.exists('./GUI_output'):
		os.mkdir('./GUI_output')

	if os.path.exists('./GUI_output/cropped_cars'):
		shutil.rmtree('./GUI_output/cropped_cars')
	os.mkdir('./GUI_output/cropped_cars')

	VMMR_model = prepare_model(config)
	# load the COCO class labels our YOLO model was trained on
	labelsPath = "./utils/yolo-coco/coco.names"
	LABELS = open(labelsPath).read().strip().split("\n")

	# derive the paths to the YOLO weights and model configuration
	weightsPath = "./utils/yolo-coco/yolov3.weights"
	configPath = "./utils/yolo-coco/yolov3.cfg"

	# load our YOLO object detector trained on COCO dataset (80 classes)
	print("[INFO] loading YOLO from disk...")
	net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

	try:
		net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
		net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
	except:
		print("YOLO cannot run with GPU. Using CPU ...")

	# determine only the *output* layer names that we need from YOLO
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	print("[INFO] accessing video stream...")
	vs = cv2.VideoCapture(video_path)
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
	total_frames = int(vs.get(prop))
	print("\n[INFO] {} total frames in video".format(total_frames))

	tracker = {"car": Sort(iou_threshold=0.1), "person": Sort(iou_threshold=0.1)}
	coords = {"car": dict(), "person": dict()}
	predictions = dict()
	for i in range(total_frames):
		predictions['car' + str(i)] = dict()
	frameIndex = 0
	pbar = tqdm(total=total_frames, position = 0)
	while True:
		pbar.update(1)
		(grabbed, frame) = vs.read()
		if not grabbed:
			break
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
				for track in tracks:
					boxes.append([track[0], track[1], track[2], track[3]])
					indexIDs.append(int(track[4]))
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
							if car.shape[0] > 64 and car.shape[1] > 64:
								predictions['car' + str(indexIDs[i])]['frame-' + str(frameIndex)] = predict(car, VMMR_model, config, 3)
								car_img.save("./GUI_output/cropped_cars/car{}_frame#{}.jpg".format(indexIDs[i], frameIndex))
							else:
								del coords[k]['frame-' + str(frameIndex)]['car' + str(indexIDs[i])]
						i += 1
		frameIndex += 1
	return predictions, coords

def create_annotated_frames(video_path, coords, final_pred):
	if os.path.exists('./GUI_output/annotated_frames'):
		shutil.rmtree('./GUI_output/annotated_frames')
	os.mkdir('./GUI_output/annotated_frames')
	if os.path.exists('./GUI_output/original_frames'):
		shutil.rmtree('./GUI_output/original_frames')
	os.mkdir('./GUI_output/original_frames')

	fps = 7
	writer = None
	(W, H) = (None, None)
	np.random.seed(1)
	COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

	frameIndex = 0
	vs = cv2.VideoCapture(video_path)
	while True:
		(grabbed, frame) = vs.read()
		if not grabbed:
			break
		if W is None or H is None:
			(H, W) = frame.shape[:2]
		cv2.imwrite(r"./GUI_output/original_frames/frame-{}.png".format(frameIndex), frame)
		if 'frame-' + str(frameIndex) in coords["car"]:
			for k, v in coords["car"]['frame-' + str(frameIndex)].items():
				x1, y1, x2, y2 = coords["car"]['frame-' + str(frameIndex)][k]
				color = [int(c) for c in COLORS[int(k[3:]) % len(COLORS)]]
				cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
				dy = 25
				for i in range(3):
					line = class_names[final_pred[k][0][-i-1]]
					yy = y1 - i * dy
					cv2.putText(frame, line, (x1, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)

		if 'frame-' + str(frameIndex) in coords["person"]:
			for k, v in coords["person"]['frame-' + str(frameIndex)].items():
				x1, y1, x2, y2 = coords["person"]['frame-' + str(frameIndex)][k]
				cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
				text = "person"
				cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
		cv2.imwrite(r"./GUI_output/annotated_frames/frame-{}.png".format(frameIndex), frame)
		# check if the video writer is None
		if writer is None:
			# initialize our video writer
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter("./GUI_output/PROCESSED_{}.avi".format(os.path.split(video_path)[1].split(".")[0]), fourcc, fps,
									 (frame.shape[1], frame.shape[0]), True)
		writer.write(frame)
		frameIndex += 1
		if frameIndex >= 400000:
			writer.release()
			vs.release()
			exit()
	writer.release()
	vs.release()