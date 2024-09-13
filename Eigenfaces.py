import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os


def loadImage(im_name):    #output: (flatten img, 30*30 img)  
    a = cv.imread(im_name, 0)
    b = cv.resize(a, dsize=(30,30))
    c = b.flatten()
    d = cv.resize(c, dsize=(1,900)) #new
    return d, b

def normal_img(a):    #output: normalized img (900,1)
    maxindex = a.argmax()
    max = a[maxindex]
    b = a/max
    b.resize((900,1))
    return b

def loadFaces(folder_name):    #output: matrix off all normalized images (900, 400)
    cwd = os.getcwd()
    os.chdir(folder_name)
    imgs = np.array(np.zeros((900,1)))
    for i in range(1, 41):
        folder = f"s{i}"
        os.chdir(folder)
        for j in range(1, 11):
            img_name = f"{j}.pgm"
            vec, _ = loadImage(img_name)
            vec = vec.flatten() #new
            n_vec = normal_img(vec)
            imgs = np.concatenate( (n_vec, imgs), axis=1)  #adds the normalized img to the main matrix.
        os.chdir(cwd)
        os.chdir(folder_name)
    os.chdir(cwd)
    imgs = imgs[:, :imgs.shape[1]-1]   #removes the last extra column (000)
    return imgs

def cov(mat):    #output: the covariance matrix (900, 900)
    cov = np.dot(mat, mat.transpose()) #(900, 400).(400, 900) = (900, 900)
    return cov/mat.shape[1]

def findEigenFaces(cov, n=0):    #output: sorted evectors and evalues of the cov matrix ((900, 25), 25)
    evals, evecs = np.linalg.eig(cov)
    ind = np.argsort(evals)  #sorted indexes of eigen values.
    if n == 0:
        ind2 = ind[:]
    else: 
        ind2 = ind[900-n :]  #the last n largest eigen values.
    ind3 = []
    for z in range(1, 1+len(ind2)):  #decreasingly sorts indexes of eigen values.
        ind3.append(ind2[len(ind2) - z])
    m = 0
    sorted_eval = np.zeros((len(ind3)), dtype=complex)
    sorted_evec = np.zeros((900, len(ind3)), dtype=complex)
    for i in ind3:
        sorted_eval[m] = evals[i]
        for j in range(900):
            sorted_evec[j, m] = evecs[j, i]
        m += 1
    return sorted_evec, sorted_eval
    
def showEigenFaces(efaces, size):
    plt.figure()
    for n in range(1, 1+size[0]*size[1]):
        ax = plt.subplot(size[0], size[1], n)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(15,-3,f"Eig {n-1}", size=7, ha='center', va='center')
        img = np.array(efaces[:, efaces.shape[1]-n], dtype=float)        #warning.
        img.resize((30,30))
        plt.imshow(img, cmap="gray")
    plt.savefig("FaceRecognition.png")
    plt.show()

def single_convertFace(x, eigenfaces):    #output: a 25 element vector of img.
    l = []
    for i in range(25):
        temp = np.array(eigenfaces[:, i])
        a = np.dot(x, temp)
        l.append(a)
    l2 = np.array(l)
    return l2

def convertFace(X, eigenfaces):    #output: matrix of all the faces turned to 25 element (25, 400)
    a = np.zeros(900)
    if X.shape == a.shape:
        return single_convertFace(X, eigenfaces)
    l = []
    for i in range(X.shape[1]):
        x = X[:, i]
        sample = single_convertFace(x, eigenfaces)
        l.append(sample)
    l2 = np.array(l)
    return l2.transpose()

def createDataset(folder_name, eigenfaces):    #output: dataset of all imgs(name, img_25_element_vec) size:(400)
    f_list = os.listdir(folder_name)
    cwd1 = os.getcwd()
    os.chdir(folder_name)
    cwd = os.getcwd()
    data = []
    for folder in f_list:
        os.chdir(folder)
        img_list = os.listdir()
        name = folder
        for img in img_list:
            img_vec, _ = loadImage(img)
            img_vec = img_vec.flatten() #new
            img_vec2 = convertFace(img_vec, eigenfaces) #turns the img to 25 element img.
            data.append((img_vec2, name)) #adds a tuple of (name, img_25_element_vec) to dataset.
        os.chdir(cwd)
    os.chdir(cwd1)
    data2 = np.array(data)
    return data2

def distance(x, y):    #output: oghlidosi(!) distance between two points.
    z = x - y
    sum = 0
    for i in range(len(z)):
        sum += z[i]*z[i]
    return pow(sum, 0.5)

def kNN(dataset, input_face_vec, eigenfaces, k):    #output: all the k nearest neighbors (name, distance) and the most repeated name.
    input_face_vec = input_face_vec.flatten()
    input = convertFace(input_face_vec, eigenfaces)
    distances = []
    data = []
    for person in dataset: #calculates the distance between input and each one of imgs in dataset and adds it to a list (name, distance)
        dist = distance(input, person[0]) 
        distances.append(dist)
        data.append((person[1], dist))
    distances = np.array(distances)
    ind = np.argsort(distances) #sorted indexes of distances.
    sorted_data = []
    for i in ind: #sorts all the persons by their distance from input.
        sorted_data.append(data[i])
    k_sorted_data = sorted_data[:k] #picks the first k persons with the least distance.
    names = [name for name, _ in k_sorted_data] 
    nn = max(set(names), key=names.count) #returns the most repeated name.
    return nn, k_sorted_data

# if __name__ == "__main__":
faces = loadFaces("att_faces")
cov_m = cov(faces)
eigf, _ = findEigenFaces(cov_m, 25)
showEigenFaces(eigf, [3,3])







    
