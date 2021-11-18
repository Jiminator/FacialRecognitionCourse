from API.FaceAPI_dlib import *
import json

class jsonEncoding(json.JSONEncoder):
	def default(self, o):
		if isinstance(o, Info):
			return {'introduce':o.other,'feature':o.ft.data} 

class Info:
	def __init__(self,name,ft,introduce='无'):
		self.name = name
		self.ft = ft
		self.other = introduce
	
	def __repr__(self):
		return "Name: %s, Gender: %s"%(self.name,self.other)

class AI:
	def __init__(self):
		self.dataset = dict()	#Save name and corresponding Info
		self.faceAPI = FaceAPI()
		self.logHistory = []
	
	def getFace(self,img):
		rects = self.faceAPI.detect(img)
		if len(rects) == 0:
			print('Input error! No face found in the image.')
			return None
		elif len(rects) > 1:
			print('Input error! The image should only contain one face')
			return None
		return rects
	
	def register(self,img,name=None,introduce='无'):
		#Register Face
		if name is None:
			print('The classmate information was not entered because the name was not provided!')
			return None
		rects = self.getFace(img)
		if rects is None:
			return None
		ft = self.faceAPI.extractFeature(img,rects[0])
		info = Info(name,ft,introduce)
		if name not in self.dataset:
			self.dataset[name] = info
			print('Successful registration! The information of this classmate is as follows:')
			print('\t\t %s'%(info))
		else:
			print('Username (%s) has been registered!'%(name))

	
	def clear(self):
		#Clear database
		self.dataset = dict()
		print('The database has been successfully emptied.')
	
	def search(self,img,thresh=0.90):
		#sign-in
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
		print('Highest Score:',det_score)
		if det_score < thresh:
			print('The classmate was not found in the database')
		else:
			print("Classmate was found in the database. The classmate's information is as follows:")
			print('\t\t %s'%(self.dataset[det_name]))
			self.logHistory.append(det_name)
		#return self.dataset[name]
	
	def log(self):
		#return sign-in info
		return self.logHistory
	
	def regNameList(self):
		#Return registration record
		res = []
		for name in self.dataset:
			res.append(name)
		return res
	
	def readCameraImg(self):
		import time
		cap = cv2.VideoCapture(0)
		time.sleep(3)
		ret, frame = cap.read()
		cap.release()
		if ret:
			return frame
		else:
			print("Read Camera Error")


	def saveDataset(self,path):
		res = json.dumps(self.dataset, sort_keys=False, indent=4,separators=(',', ': '),cls=jsonEncoding)
		with open(path,'w') as f:
			f.write(res)
		
	def loadDataset(self,path):
		with open(path,'r') as f:
			text = f.read()
		res = json.loads(text)
		self.dataset = dict()
		for name in res:
			feature = Feature(res[name]['feature'])
			info = Info(name,feature,res[name]['introduce'])
			self.dataset[name] = info

		

if __name__ == '__main__':
	ai = AI()


