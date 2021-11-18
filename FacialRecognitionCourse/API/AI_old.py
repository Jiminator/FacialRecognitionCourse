# from API.FaceAPI_dlib import *
# import cv2

class Info:
	def __init__(self,name,ft,introduce='无'):
		self.name = name
		self.ft = ft
		self.other = introduce
	
	def __repr__(self):
		return "姓名: %s 注册信息: %s"%(self.name,self.other)

class AI:
	def __init__(self):
		self.dataset = dict()	#储存姓名和对应Info
		self.faceAPI = FaceAPI()
		self.logHistory = []
	
	def getFace(self,img):
		rects = self.faceAPI.detect(img)
		if len(rects) == 0:
			print('输入错误! 未在图像中找到人脸')
			return None
		elif len(rects) > 1:
			print('输入错误！图像中应只包含一个人脸')
			return None
		return rects
	
	def register(self,img,name=None,introduce='无'):
		if name is None:
			print('因未提供名称，该同学信息未被录入!')
			return None
		rects = self.getFace(img)
		if rects is None:
			return None
		ft = self.faceAPI.extractFeature(img,rects[0])
		info = Info(name,ft,introduce)
		if name not in self.dataset:
			self.dataset[name] = info
			print('成功注册! 该同学的信息如下:')
			print('\t\t %s'%(info))
		else:
			print('用户名称(%s)已被注册!'%(name))

	
	def clear(self):
		self.dataset = dict()
		print('已成功清空数据库!')
	
	def search(self,img,thresh=0.90):
		rects = self.getFace(img)
		if rects is None:
			return None
		ft = self.faceAPI.extractFeature(img,rects[0])
		det_score = -1
		det_name = None
		for name in self.dataset:
			score = self.faceAPI.compare(ft,self.dataset[name].ft,return_score=True)
			if score > det_score:
				det_score = score
				det_name = name
		print('最高分数:',det_score)
		if det_score < thresh:
			print('未在数据库中找到该同学')
			return None
		else:
			print('找到该同学! 该同学的信息如下:')
			print('\t\t %s'%(self.dataset[det_name]))
			self.logHistory.append(det_name)
			return det_name
	
	def log(self):
		return self.logHistory
	
	def regNameList(self):
		res = []
		for name in self.dataset:
			res.append(name)
		return res

	def readCameraImg(self):
		cap = cv2.VideoCapture(0)
		ret, frame = cap.read()
		cap.release()
		if ret:
			return frame
		else:
			print("Read Camera Error")




import cv2
im = cv2.imread('../data/0.png') # 读取目录下的jpg图像
cv2.imshow('image', img) # 建立名为‘image’ 的窗口并显示图像
k = cv2.waitKey(0) # waitkey代表读取键盘的输入，括号里的数字代表等待多长时间，单位ms。 0代表一直等待

'''
if __name__ == '__main__':
	ai = AI()
'''
	
		

			


