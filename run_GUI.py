import os
from tkinter import *
from tkinter import filedialog

from utils.Color_Recognition import color_recognition
from utils.Generate_db import generate_db
from utils.Detect_and_Track import *

import warnings
warnings.simplefilter("ignore", UserWarning)


def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def make_video(database, topk, idx, fps, coords, final_pred, predictions):
    if not os.path.exists("./GUI_output/extracted_clips"):
        os.mkdir("./GUI_output/extracted_clips")
    entry = int(np.round(database.iloc[idx]['entry'].total_seconds()*fps))
    exit = int(np.round(database.iloc[idx]['exit'].total_seconds()*fps))
    video_name = '_'.join(
        [database.iloc[idx]['make'], database.iloc[idx]['model'], database.iloc[idx]['year']])
    car_nb = list(predictions.keys())[idx]
    image_folder = r'./GUI_output/original_frames'
    video = None
    for img in sorted(os.listdir(image_folder), key=numericalSort):
        ind = int(img.split('.')[0][6:])
        if ind >= entry and ind <= exit:
            frame = cv2.imread(os.path.join(image_folder, img))
            for car in coords["car"]['frame-' + str(ind)]:
                if class_names[final_pred[car][0][topk]] == video_name:
                    break
            if class_names[final_pred[car][0][topk]] != video_name:
                print("ERROR, cannot find the vehicle {}".format(video_name))
            x1, y1, x2, y2 = coords["car"]['frame-' + str(ind)][car]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            dy = 25
            for i in range(3):
                line = class_names[final_pred[car][0][-i - 1]]
                yy = y1 - i * dy
                cv2.putText(frame, line, (x1, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            height, width, layers = frame.shape
            if video == None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                video = cv2.VideoWriter('./GUI_output/extracted_clips/{}.avi'.format(video_name.replace(" ", "_")), fourcc,
                                        fps, (width, height), True)
            video.write(frame)
    video.release()
    os.system(".\GUI_output\extracted_clips\{}.avi".format(video_name.replace(" ", "_")))

def click():
    global video_path
    video_path = filedialog.askopenfilename(initialdir="./test_videos")
    label = Label(main_frame, text=os.path.split(video_path)[1])
    label.grid(row=2, column=0, pady=5, sticky=W)

def play_video(video_path):
    os.system(".\GUI_output\PROCESSED_{}.avi".format(os.path.split(video_path)[1].split(".")[0]))

def intersect(*d):
    sets = iter(map(set, d))
    result = sets.next()
    for s in sets:
        result = result.intersection(s)
    return result

def search(database, fps, entry_make, entry_model, entry_year, entry_colour, coords, final_pred, predictions):
    global widgets
    criteria = {"make": str(entry_make.get()),
                "model": str(entry_model.get()),
                "year":  str(entry_year.get()),
                "colour": str(entry_colour.get())}
    criteria = {k: v for k, v in criteria.items() if v!=""}
    idx = dict()
    for i in range(3):
        idx[i] = []
        for c in criteria:
            idx[i].append(list(database[i].loc[database[i][c] == criteria[c]].index))
        idx[i] = set(idx[i][0]).intersection(*idx[i])
    total_ = len(idx[0])+len(idx[1])+len(idx[2])
    label_query_result = Label(main_frame, text="There are {} cars in the video that match your descriptions".format(total_))
    label_query_result.grid(row=11, column=0, sticky=W, padx=20, pady=8, columnspan=2)

    top1 = Label(main_frame, text="Top 1 matches")
    top1.grid(row=4, column=0, sticky=W, padx=400, pady=30, columnspan=2)

    top2 = Label(main_frame, text="Top 2 matches")
    top2.grid(row=4, column=0, sticky=W, padx=600, pady=30, columnspan=2)

    top3 = Label(main_frame, text="Top 3 matches")
    top3.grid(row=4, column=0, sticky=W, padx=800, pady=30, columnspan=2)

    try:
        for widget in widgets:
            widget.destroy()
    except:
        pass

    x = 400
    widgets = []
    for i in range(3):
        r = 5
        for j in idx[i]:
            car_name = "_".join([database[i].iloc[j].make, database[i].iloc[j].model, database[i].iloc[j].year])
            car_label = Label(main_frame, text=car_name)
            car_label.grid(row=r, column=0, sticky=W, padx=x, pady=8, columnspan=2)
            extract = Button(main_frame, text="View", command=(lambda i=i, j=j: make_video(database[i],i, j, fps, coords, final_pred, predictions)), height=1, width=3)
            extract.grid(row=r, column=0, sticky=W, padx=x+150, pady=8, columnspan=2)
            r += 1
            widgets.append(car_label)
            widgets.append(extract)
        x += 200

def main():
    processing = Label(main_frame, text="Processing ... This can take few minutes. Please wait.")
    processing.grid(row=1, column=0, sticky=W, padx=600, pady=5, columnspan=2)
    processing_text = Text(main_frame, height=2, width=100)
    processing_text.grid(row=2, column=0, padx=250, pady=5, sticky=W, columnspan=2)

    processing_text.insert(END, '    Detection & Tracking ...')
    print("[INFO] Detection & Tracking ...")
    root.update()
    predictions, BBs_coordinates = detect_track(video_path)
    final_predictions = merge_predictions(predictions)
    create_annotated_frames(video_path, BBs_coordinates, final_predictions)

    processing_text.insert(END, 'Color recognition ...')
    print("\n[INFO] Color recognition ...")
    root.update()
    car_colours = color_recognition()

    processing_text.insert(END, 'Creating annotated video ...')
    print("[INFO] Creating annotated video ...")
    root.update()
    play = Button(main_frame, text="Play processed video", command=(lambda: play_video(video_path)))
    play.grid(row=3, column=0, pady=5, padx=590, sticky=W)

    processing_text.insert(END, 'Generating database ...')
    print("[INFO] Generating database ...")
    root.update()
    database, fps = generate_db(video_path, predictions, final_predictions, car_colours)

    processing.destroy()
    processing = Label(main_frame, text="Processing completed successfully !")
    processing.grid(row=1, column=0, sticky=W, padx=600, pady=5, columnspan=2)

    query = Label(main_frame, text="Query")
    query.grid(row=4, column=0, sticky=W, padx=130, pady=30, columnspan=2)



    label_make = Label(main_frame, text="Make")
    label_make.grid(row=5, column=0, sticky=W, padx=20, pady=8, columnspan=2)
    entry_make = Entry(main_frame)
    entry_make.grid(row=5, column=0, sticky=W, padx=80, pady=8, columnspan=2)

    label_model = Label(main_frame, text="Model")
    label_model.grid(row=6, column=0, sticky=W, padx=20, pady=8, columnspan=2)
    entry_model = Entry(main_frame)
    entry_model.grid(row=6, column=0, sticky=W, padx=80, pady=8, columnspan=2)

    label_model = Label(main_frame, text="Year")
    label_model.grid(row=7, column=0, sticky=W, padx=20, pady=8, columnspan=2)
    entry_year = Entry(main_frame)
    entry_year.grid(row=7, column=0, sticky=W, padx=80, pady=8, columnspan=2)

    label_model = Label(main_frame, text="Colour")
    label_model.grid(row=8, column=0, sticky=W, padx=20, pady=8, columnspan=2)
    entry_colour = Entry(main_frame)
    entry_colour.grid(row=8, column=0, sticky=W, padx=80, pady=8, columnspan=2)

    play = Button(main_frame, text="Search", command=(lambda: search(database, fps, entry_make, entry_model, entry_year, entry_colour, BBs_coordinates, final_predictions, predictions)))
    play.grid(row=9, column=0, sticky=W, padx=130, pady=8, columnspan=2)

root = Tk()
root.geometry('1300x850')
root.title("Vehicle Make and Model Recognition")

main_frame = Frame(root)
main_frame.grid(row=0)

browse = Button(main_frame, text ="Browse your video", command=click)
browse.grid(row=1, column=0, pady=5, sticky=W)

upload = Button(main_frame, text ="Upload your video", command=main)
upload.grid(row=3, column=0, pady=5, sticky=W)

root.mainloop()