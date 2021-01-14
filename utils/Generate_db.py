import pandas as pd
import cv2
import datetime

def generate_db(video_path, predictions, final_pred, car_color):
    predictions = {k: v for k, v in predictions.items() if v}
    class_names = []
    with open(r"./data/class_names_2522c.txt", 'r') as f:
        for line in f:
            class_names.append(line.rstrip('\n'))

    date_in = []
    date_out = []
    make = {0:[], 1:[], 2:[]}
    model = {0:[], 1:[], 2:[]}
    year = {0:[], 1:[], 2:[]}
    color = []
    database = dict()

    v = cv2.VideoCapture(video_path)
    fps = v.get(cv2.CAP_PROP_FPS)
    n = 1/fps
    for car, frame in predictions.items():
        date_in.append(datetime.timedelta(seconds=int(list(frame.keys())[0].split('-')[-1])*n))
        date_out.append(datetime.timedelta(seconds=int(list(frame.keys())[-1].split('-')[-1])*n))
        for i in range(3):
            pred = class_names[final_pred[car][0][i]]
            make[i].append(pred.split('_')[0])
            model[i].append(' '.join(pred.split('_')[1:-1]))
            year[i].append(pred.split('_')[-1])
        color.append(car_color[car])
    for i in range(3):
        database[i] = pd.DataFrame(
            columns=['entry', 'exit', 'make', 'model', 'year', 'colour'])
        database[i]['entry'] = date_in
        database[i]['exit'] = date_out
        database[i]['make'] = make[i]
        database[i]['model'] = model[i]
        database[i]['year'] = year[i]
        database[i]['colour'] = color

    return database, fps
