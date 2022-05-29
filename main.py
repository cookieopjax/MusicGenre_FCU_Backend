import operator
import os
import pickle
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

# --- using the model ---
dataset = []
results = ['', 'blues', 'classical', 'country', 'disco',
           'hippop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
isConvert = False


def loadDataset(filename):
    with open("mydataset.dat", 'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break


loadDataset("mydataset.dat")


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


def convertToMp3(fileUrl):
    dst = "convertedMp3.wav"
    audSeg = AudioSegment.from_mp3(fileUrl)
    audSeg.export(dst, format="wav")
    return dst


def checkType(fileName):
    audType = filetype.guess(fileName)

    if (audType != None):
        if (audType.mime == "audio/mpeg"):
            inputFile = convertToMp3(fileName)
            isConvert = True
        elif (audType.mime == "audio/x-wav"):
            inputFile = fileName
            isConvert = False

    return inputFile


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    content = await file.read()
    f = open(file.filename, "wb")
    f.write(content)

    inputFile = checkType(file.filename)

    print(inputFile)
    (rate, sig) = wav.read(inputFile)
    mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
    covariance = np.cov(np.matrix.transpose(mfcc_feat))
    mean_matrix = mfcc_feat.mean(0)
    feature = (mean_matrix, covariance, 0)

    pred = nearestClass(getNeighbors(dataset, feature, 5))

    f.close()
    os.remove(file.filename)
    if(isConvert):
        os.remove(inputFile)

    return {"genre": results[pred]}
