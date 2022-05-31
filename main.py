from email import message
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
    (rate, sig) = wav.read(fileName)
    mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
    covariance = np.cov(np.matrix.transpose(mfcc_feat))
    mean_matrix = mfcc_feat.mean(0)
    feature = (mean_matrix, covariance, 0)

    pred = nearestClass(getNeighbors(dataset, feature, 5))

    return results[pred]

 # handle type convert


def convertToMp3(fileUrl):
    dst = "convertedMp3.wav"
    audSeg = AudioSegment.from_mp3(fileUrl)
    audSeg.export(dst, format="wav")
    return dst


def typeHandler(fileName):
    audType = filetype.guess(fileName)
    global isConvert

    if (audType != None):
        if (audType.mime == "audio/mpeg"):
            tempFile = convertToMp3(fileName)
            isConvert = True
        elif (audType.mime == "audio/x-wav"):
            tempFile = fileName
            isConvert = False

    return tempFile


@app.get("/")
def read_root():
    return {"status": "ok"}


@app.post("/uploadfile/")
async def gnereDetection(file: UploadFile):
    # read file and save in server
    content = await file.read()
    f = open(file.filename, "wb")
    f.write(content)
    global originFile
    originFile = file.filename

    # the audio length should under 60s
    if(AudioSegment.from_file(originFile).duration_seconds > 61):
        f.close()
        os.remove(originFile)
        return {"genre": "", "status": "error", "message": "音檔長度不可高於60秒"}

    # if mp3 convert it to wav
    global convertedFile
    convertedFile = typeHandler(originFile)
    f.close()
    print('已讀取以及轉換檔案 : ' + convertedFile)

    return {"status": "the file is get ready"}


@app.get("/predict/")
def predictRoute():
    print('開始預測 : ' + convertedFile)
    # start to predict the audio genre
    genre = predict(convertedFile)

    # delete these file
    os.remove(originFile)
    if(isConvert):
        os.remove(convertedFile)

    return {"genre": genre, "status": "ok"}
