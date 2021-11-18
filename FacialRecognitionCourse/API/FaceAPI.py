#version 1: dlib based

import numpy as np

class Point:
	def __init__(self,x,y):
		self.x = x
		self.y = y

class Rect:
	def __init__(self,x1,y1,x2,y2):
		self.p1 = Point(x1,y1)
		self.p2 = Point(x2,y2)


class FaceAPI:
	def __init__(self):
		self.detectModel = None	
		self.alignModel = None	
		self.verifyModel = None	

	def loadDetectModel(self):
		pass
	
	def loadVerifyModel(self):
		pass
	
	def loadAlignModel(self):
		pass

	def detect(self,img):
		#img[ndarray]:[H,W,3] or [H,W,1]
		#return : list(Rect)
		pass
	
	def align(self,img,rect):
		#img[ndarray]:[H,W,3] or [H,W,1]
		#rect[Rect]
		#return : np.array [N,2] 
		pass

	def extractFeature(self,img,rect):
		#img[ndarray]:[H,W,3] or [H,W,1]
		#rect[Rect]
		#return : np.array [L]
		pass
	
	def compare(self,ftA,ftB):
		#ftA/B: np.array [L]
		#return[bool]
		pass
		
		


