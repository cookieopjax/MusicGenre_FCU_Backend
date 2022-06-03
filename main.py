from email import message
from hashlib import new
import operator
import os
import pickle
from email.mime import audio
from tempfile import TemporaryFile
from wsgiref.validate import InputWrapper

import filetype
import numpy as np
import scipy.io.wavfile as wav
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
from python_speech_features import mfcc

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


dataset = []
results = ['', 'blues', 'classical', 'country', 'disco',
           'hippop', 'jazz', 'metal', 'pop', 'reggae', 'rock']


def loadDataset(filename):
    with open(filename, 'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break


loadDataset("mydataset.dat")


# machine learing method

def distance(instance1, instance2, k):
    distance = 0
    mm1 = instance1[0]
    cm1 = instance1[1]
    mm2 = instance2[0]
    cm2 = instance2[1]
    distance = np.trace(np.dot(np.linalg.inv(cm2), cm1))
    distance += (np.dot(np.dot((mm2-mm1).transpose(),
                 np.linalg.inv(cm2)), mm2-mm1))
    distance += np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    distance -= k
    return distance


def getNeighbors(trainingSet, instance, k):
    distances = []
    for x in range(len(trainingSet)):
        dist = distance(trainingSet[x], instance, k) + \
            distance(instance, trainingSet[x], k)
        distances.append((trainingSet[x][2], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def nearestClass(neighbors):
    classVote = {}
    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVote:
            classVote[response] += 1
        else:
            classVote[response] = 1
    sorter = sorted(classVote.items(),
                    key=operator.itemgetter(1), reverse=True)
    return sorter[0][0]


def predict(fileName):

    print('開始預測 : ' + fileName)
    (rate, sig) = wav.read(fileName)
    mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
    covariance = np.cov(np.matrix.transpose(mfcc_feat))
    mean_matrix = mfcc_feat.mean(0)

    feature = (mean_matrix, covariance, 0)
    print("start pred")
    pred = nearestClass(getNeighbors(dataset, feature, 5))
    print("end nearest")

    return results[pred]

 # handle type convert


def convertToMp3(fileUrl, dst):
    audSeg = AudioSegment.from_mp3(fileUrl)
    audSeg.export(dst, format="wav")


def typeHandler(fileName):
    # 如果檔案為a.wav，其檔名頭加上new_a.wav
    # 如果檔案為a.mp3, 轉為 new_a.wav，並刪除a.mp3

    audType = filetype.guess(fileName)
    newFile = 'new_' + fileName.split('.')[0] + '.wav'

    # 如果要轉換的目標檔案已存在，就直接return
    if(os.path.isfile(newFile)):
        return fileName

    if (audType != None):
        if (audType.mime == "audio/mpeg"):
            convertToMp3(fileName, newFile)
            os.remove(fileName)

        elif (audType.mime == "audio/x-wav"):
            os.rename(fileName, newFile)

    return newFile


@app.get("/")
def read_root():
    return {"status": "ok"}


@app.post("/uploadfile/")
async def uploadFile(file: UploadFile):
    # read file and save in server
    content = await file.read()
    f = open(file.filename, "wb")
    f.write(content)
    global originFile
    originFile = file.filename
    f.close()

    # the audio length should under 60s
    if(AudioSegment.from_file(originFile).duration_seconds > 61):
        f.close()
        os.remove(originFile)
        return {"genre": "", "status": "error", "message": "音檔長度不可高於60秒"}

    # if mp3 convert it to wav

    convertedFile = typeHandler(originFile)

    if(os.path.isfile(convertedFile)):
        print('已讀取以及轉換檔案 : ' + convertedFile)

    return {"convertedFile": convertedFile, "status": "the file is get ready"}


@app.get("/predict/")
def predictRoute(fileName: str):
    # start to predict the audio genre
    print('fileName:' + fileName)

    genre = predict(fileName)

    # delete these file
    print("檔案移除 ! ")
    os.remove(fileName)

    return {"genre": genre, "status": "ok"}
