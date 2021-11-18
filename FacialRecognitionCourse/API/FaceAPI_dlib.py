#dlib based
import numpy as np
import dlib
import numpy as np
import cv2
import matplotlib.pyplot as plt

path_align_model = 'API/shape_predictor_5_face_landmarks.dat'
path_verify_model = 'API/dlib_face_recognition_resnet_model_v1.dat'



def loadImage(path):
	img = cv2.imread(path)
	if img is None:
		print('Error: Please check the path.')
	return img

class Point:
	def __init__(self,x,y):
		self.x = x
		self.y = y

	def __repr__(self):
		return "Point(%d,%d)"%(self.x,self.y)

class Rect:
	def __init__(self,x1,y1,x2,y2):
		self.p1 = Point(x1,y1)
		self.p2 = Point(x2,y2)
	
	def __repr__(self):
		return "Rect(%d,%d,%d,%d)"%(self.p1.x,self.p1.y,self.p2.x,self.p2.y)

class Feature:
	def __init__(self,data=None):
		self.data = data
		self.lenght = len(data)

	def __repr__(self):
		tmp = ['%.2f'%(t) for t in self.data]
		return "Feature(%s)"%(','.join(list(tmp)))


def drect2rect(drect):
	rect = Rect(drect.left(),drect.top(),drect.right(),drect.bottom())
	return rect

def rect2drect(rect):
	drect = dlib.rectangle(rect.p1.x,rect.p1.y,rect.p2.x,rect.p2.y)
	return drect

def dpoint2point(dpoint):
	point = Point(dpoint.x,dpoint.y)
	return point

def point2dpoint(point):
	dpoint = dlib.point(point.x,point.y)
	return dpoint

def dft2ft(dft):
	data = []
	for tmp in dft:
		data.append(tmp)
	ft = Feature(data)
	return ft

class FaceAPI:
	def __init__(self):
		self.detectModel = dlib.get_frontal_face_detector() 
		self.alignModel = None	
		self.verifyModel = None	
		self.loadAlignModel(path_align_model)
		self.loadVerifyModel(path_verify_model)

	def loadDetectModel(self,path):
		pass
	
	def loadVerifyModel(self,path):
		self.verifyModel = dlib.face_recognition_model_v1(path)
	
	def loadAlignModel(self,path):
		self.alignModel = dlib.shape_predictor(path)

	def detect(self,img):
		#img[ndarray]:[H,W,3] or [H,W,1]
		#return : list(Rect)
		gray = None
		if len(img.shape) == 2:
			gray = img
		else:
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		dets = self.detectModel(gray, 1)	#list(dlib.rect)
		res = list()
		for drect in dets:
			res.append(drect2rect(drect))
		return res

	def align(self,img,rect):
		#img[ndarray]:[H,W,3] or [H,W,1]
		#rect[Rect]
		#return :  list(Point)
		drect = rect2drect(rect)
		lms = self.alignModel(img, drect)
		res = list()
		for dpoint in lms.parts():
			res.append(dpoint2point(dpoint))
		return res

	def extractFeature(self,img,rect):
		#img[ndarray]:[H,W,3] or [H,W,1]
		#rect[Rect]
		#return : np.array [L]
		drect = rect2drect(rect)
		lms = self.alignModel(img, drect)
		feature = self.verifyModel.compute_face_descriptor(img, lms)
		res = dft2ft(feature)
		return res
	
	def compare(self,ftA,ftB,thresh=0.90,return_score=False):
		#ftA/B: Feature
		#return[bool]
		a = np.array(ftA.data)
		b = np.array(ftB.data)
		score = ((a*b).sum())/(np.sqrt((a*a).sum())*np.sqrt((b*b).sum()))
		if return_score:
			return score
		if score > thresh:
			return True
		else:
			return False

def show(img):
	plt.figure()
	plt.imshow(img[:,:,::-1])
	plt.show()

def showWithRects(img,rects):
	for rect in rects:
		cv2.rectangle(img,(rect.p1.x,rect.p1.y),(rect.p2.x,rect.p2.y),(0,0,255),5)
	show(img)
		


