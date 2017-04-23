import time
import math
import numpy as np
from sklearn import svm
import os, sys, pickle, wave
from pydub import AudioSegment
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel

# path = os.getcwd() + '/pyAudioAnalysis/'
# if path not in sys.path: sys.path.append(path)

emotions = {'angry': 0,
            'happy': 1,
            'unhappy': 2,
            'neutral': 3}
target_names = emotions.keys()
labels = emotions.values()

# Function to create a dictype object to map input variables with target variables
def create_dict(files, directory, emotion_value='None'):
    local_files = filter(lambda x: x[-4:] == '.mp3' or x[-4:] == '.aac', os.listdir(directory))
    local_files = map(lambda x: directory+'/'+x, local_files)
    for filename in local_files:
        files[filename] = emotion_value


# Function to transform input data into feature set
def get_features_1(file_list):
    X = []
    for filename in file_list:
        temp = []
        if filename[-4:] == '.mp3':
            sound = AudioSegment.from_mp3(filename)
        elif filename[-4:] == '.aac':
            sound = AudioSegment.from_file(filename, format='aac')
        sound.export('test.wav',format="wav")
        [Fs, x] = audioBasicIO.readAudioFile('test.wav')
        F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.05*Fs, 0.025*Fs)
        for i in range(len(F)):
            temp.append(np.mean(F[i]))
            temp.append(np.std(F[i]))
            temp.append(max(F[i]))
            temp.append(min(F[i]))
        X.append(temp)
        # print(filename, len(y), sr, len(std_dev(mfcc)))
    return X


# Function to split dataset into train/test data given a split ratio
def partition(files, test_size):
    filenames = np.array([filename for filename in files])
    X_train, X_test = train_test_split(filenames, test_size=test_size)
    return X_train, X_test


# Transform dataset and evaluate model performance based on cross validation score
def process(classifier_object, files, feature_function):
    # np.random.seed(0)
    training_files, test_files = partition(files, 0.33)
    X = feature_function(training_files)
    y = np.array(map(lambda x: files[x], training_files))
    classifier_object.fit(X, y)
    # print('Prediction')
    X = feature_function(test_files)
    predicted = classifier_object.predict(X)
    expected = map(lambda x: files[x], test_files)
    print('Cross-validation score - ', classifier_object.score(X, expected))
    return expected, predicted


# Function to write model score into a file
def write_to_file(time_diff, classifier_object, feature_function, result):
    filename = str(classifier_object.__class__.__name__)+'_'+str(feature_function.__name__)+'.txt'
    f = open(filename, 'a')
    f.write('Time taken ='+str(time_diff)+'min \n')
    f.write(result)
    f.close()

# Function to write model score into standard console
def write_to_console(time_diff, classifier_object, feature_function, result):
    print('Time taken ='+str(time_diff)+'min')
    print(classifier_object, feature_function)
    print(result)


# Wrapper around process function
def perform(classifier_object, files, feature_function):
    start = time.time()
    expected, predicted = process(classifier_object, files, feature_function)
    end = time.time()
    time_diff = (end-start)/60
    # result = metrics.classification_report(expected, predicted, labels, target_names)
    # write_to_file(time_diff, classifier_object, feature_function, result)
    # write_to_console(time_diff, classifier_object, feature_function, result)


# Function to train a classifier and save it in a file for later usage
def train_classifier(classifier_object, train_files, feature_function):
    X = np.array(feature_function(train_files))
    y = np.array(map(lambda x: files[x], train_files))
    # clf = ExtraTreesClassifier()
    # clf = clf.fit(X, y)
    # X_new = model.transform(X)
    classifier_object.fit(X, y)
    # Save model in a file for making predictions later
    f = open('voice_files_classifier.pickle', 'wb')
    pickle.dump(classifier_object, f)
    f.close()


# Load model from the disk and make predictions
def make_predictions(test_files, feature_function):
    X_test = np.array(feature_function(test_files))
    f = open('voice_files_classifier.pickle', 'rb')
    model = pickle.load(f)
    # Now we can make predictions for this file
    predictions = model.predict_proba(X_test)
    # List containing the Probabilities for unhappy and happy respectively
    predictions = map(lambda x: [x[0]+x[2], x[1]+x[3]], predictions)
    return predictions


if __name__ == '__main__':
    files = {}
    path = './hack-the-talk-exotel-master/training_dataset/'
    for name, value in emotions.iteritems():
        create_dict(files, path+name, value)

    test_files = {}
    path = './VoiceFiles'
    create_dict(test_files, path)
    
    # classifier_object = svm.SVC()
    classifier_object = RandomForestClassifier(n_estimators=100, bootstrap=True)
    # perform(classifier_object, files, get_features_1)

    # Probabilities for each possible class
    train_classifier(classifier_object, files, get_features_1)
    predictions = make_predictions(test_files, get_features_1)
    
    print predictions
