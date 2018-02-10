import csv
import numpy as np
import sys
import time
import math 

start_time=time.time()											#exec time
np.set_printoptions(threshold=1000)								#print threshold for NumPy arrays etc.

vocabularyText=sys.argv[1]										#commandLine arguments for filenames
classMap=sys.argv[2]
training_label=sys.argv[3]
training_data=sys.argv[4]
testing_label=sys.argv[5]
testing_data=sys.argv[6]
testOn=sys.argv[7]
estimatorType=sys.argv[8]

classData=np.zeros((11269,2))
predictedLabel=np.zeros((11269,2))
testClassData=np.zeros((7505,2))
testPredictedLabel=np.zeros((7505,2))
classDocuments=[]
testClassDocuments=[]
classPriori=np.zeros((20,1))
wordData=np.zeros((2000000,3))
testWordData=np.zeros((1000000,3))
classwiseCorrectlyClassified=np.zeros((20,1))
vocabulary=[]
totalWords=0
file1 = open(vocabularyText,"r+")								#reading Vocab file
totalWords=len(file1.readlines())
file1.seek(0)
		
for i in range(totalWords):										#constructing Vocab list
	y=file1.readline()
	vocabulary.append(y)
			

#print(vocabulary)
#print(totalWords)

file1.close()


wordCount=np.zeros((totalWords+1,21))							#empty public arrays to be used Later 
classTotalWords=np.zeros(21)
PMLE=np.zeros((totalWords+1,21))
PBE=np.zeros((totalWords+1,21))

def learnNaiveBayesText():										#learning function
	
	with open(training_label) as csvfile1:						#reading training_label csv file
		readCSV1 = csv.reader(csvfile1, delimiter=',')
		n=0
		for row in readCSV1:
			classData[n,1]=n+1									#Constructing array of docID vs ClassID
			classData[n,0]=int(row[0])
			n+=1
		csvfile1.close()
		#print(classData)
		#print(classData[:,0])
		#print(classData[:481])
		
	column_1=classData[:,0]										#Slicing only classIDs
		
		
	for x in range(20):
		L=[]
		classDocuments.append(L)								#constructing empty list of lists to contain class-wise DocIDs
		
		

	g=0	
	for x in list(column_1):									#class-wise seperated lists of document names									
		if(x==1):classDocuments[0].append(g+1)
		elif(x==2):classDocuments[1].append(g+1)
		elif(x==3):classDocuments[2].append(g+1)
		elif(x==4):classDocuments[3].append(g+1)
		elif(x==5):classDocuments[4].append(g+1)
		elif(x==6):classDocuments[5].append(g+1)
		elif(x==7):classDocuments[6].append(g+1)
		elif(x==8):classDocuments[7].append(g+1)
		elif(x==9):classDocuments[8].append(g+1)
		elif(x==10):classDocuments[9].append(g+1)
		elif(x==11):classDocuments[10].append(g+1)
		elif(x==12):classDocuments[11].append(g+1)
		elif(x==13):classDocuments[12].append(g+1)
		elif(x==14):classDocuments[13].append(g+1)
		elif(x==15):classDocuments[14].append(g+1)
		elif(x==16):classDocuments[15].append(g+1)
		elif(x==17):classDocuments[16].append(g+1)
		elif(x==18):classDocuments[17].append(g+1)
		elif(x==19):classDocuments[18].append(g+1)
		elif(x==20):classDocuments[19].append(g+1)
		g+=1;
		
	
		

	for x in range(20):
		classPriori[x]=float(len(classDocuments[x]))/11269		#class Prior probabilities
		S = 'P(Omega='+repr(x+1)+') = '+str(classPriori[x])		#Printing class Priors
		print(S)

	#print(classDocuments)
	#print(np.sum(classDocuments,axis=0))


	with open(training_data) as csvfile2:						#opening training_data csv file
		readCSV2 = csv.reader(csvfile2, delimiter=',')
		m=0
		for row in readCSV2:
			wordData[m,0]=int(row[0])							#array of docID vs wordID vs wordCount
			wordData[m,1]=int(row[1])
			wordData[m,2]=int(row[2])
			m+=1
				
		csvfile2.close()


	
		
		
	#print(wordData)
	
	
	L1=0
	while(int(wordData[L1,0])in list(classDocuments[0])):		#if the doc read from first column of train_data is in class 1, put the corresponding 
		wordCount[int(wordData[L1,1]),1]+=wordData[L1,2]		#word Wi (actually its count) in the (i,1)th cell of the wordCount array Easier to index
		L1+=1;													#because of the exact index and easy computation of estimators facilitated

	L2=L1
	while(int(wordData[L2,0])in list(classDocuments[1])):		#similarly for class 2
		wordCount[int(wordData[L2,1]),2]+=wordData[L2,2]
		L2+=1;

	L3=L2
	while(int(wordData[L3,0])in list(classDocuments[2])):		#and so on...
		wordCount[int(wordData[L3,1]),3]+=wordData[L3,2]
		L3+=1;

	L4=L3
	while(int(wordData[L4,0])in list(classDocuments[3])):
		wordCount[int(wordData[L4,1]),4]+=wordData[L4,2]
		L4+=1;
		
	L5=L4
	while(int(wordData[L5,0])in list(classDocuments[4])):
		wordCount[int(wordData[L5,1]),5]+=wordData[L5,2]
		L5+=1;

	L6=L5
	while(int(wordData[L6,0])in list(classDocuments[5])):
		wordCount[int(wordData[L6,1]),6]+=wordData[L6,2]
		L6+=1;

	L7=L6
	while(int(wordData[L7,0])in list(classDocuments[6])):
		wordCount[int(wordData[L7,1]),7]+=wordData[L7,2]
		L7+=1;
		
	L8=L7
	while(int(wordData[L8,0])in list(classDocuments[7])):
		wordCount[int(wordData[L8,1]),8]+=wordData[L8,2]
		L8+=1;
		
	L9=L8
	while(int(wordData[L9,0])in list(classDocuments[8])):
		wordCount[int(wordData[L9,1]),9]+=wordData[L9,2]
		L9+=1;
		
	L10=L9
	while(int(wordData[L10,0])in list(classDocuments[9])):
		wordCount[int(wordData[L10,1]),10]+=wordData[L10,2]
		L10+=1;
		
	L11=L10
	while(int(wordData[L11,0])in list(classDocuments[10])):
		wordCount[int(wordData[L11,1]),11]+=wordData[L11,2]
		L11+=1;
		
	L12=L11
	while(int(wordData[L12,0])in list(classDocuments[11])):
		wordCount[int(wordData[L12,1]),12]+=wordData[L12,2]
		L12+=1;
		
	L13=L12
	while(int(wordData[L13,0])in list(classDocuments[12])):
		wordCount[int(wordData[L13,1]),13]+=wordData[L13,2]
		L13+=1;
		
	L14=L13
	while(int(wordData[L14,0])in list(classDocuments[13])):
		wordCount[int(wordData[L14,1]),14]+=wordData[L14,2]
		L14+=1;
		
	L15=L14
	while(int(wordData[L15,0])in list(classDocuments[14])):
		wordCount[int(wordData[L15,1]),15]+=wordData[L15,2]
		L15+=1;
		
	L16=L15
	while(int(wordData[L16,0])in list(classDocuments[15])):
		wordCount[int(wordData[L16,1]),16]+=wordData[L16,2]
		L16+=1;
		
	L17=L16
	while(int(wordData[L17,0])in list(classDocuments[16])):
		wordCount[int(wordData[L17,1]),17]+=wordData[L17,2]
		L17+=1;
		
	L18=L17
	while(int(wordData[L18,0])in list(classDocuments[17])):
		wordCount[int(wordData[L18,1]),18]+=wordData[L18,2]
		L18+=1;
		
	L19=L18
	while(int(wordData[L19,0])in list(classDocuments[18])):
		wordCount[int(wordData[L19,1]),19]+=wordData[L19,2]
		L19+=1;

	L20=L19													
	while(int(wordData[L20,0])in list(classDocuments[19])):										#for class 20
		wordCount[int(wordData[L20,1]),20]+=wordData[L20,2]
		L20+=1;

	#print(wordCount[1900,10])

	
	classTotalWords=np.sum(wordCount, axis=0)													#list of totalWords in each class 'n'
	#print(classTotalWords)

	
	for i in range(totalWords+1):
		for j in range(21):
			if(int(classTotalWords[j])==0):PMLE[i,j]=0											#to avoid DivideByZero error 
			else: PMLE[i,j]=float(wordCount[i,j])/int(classTotalWords[j])						#nk/n
	
	#print("PMLE")
	#print(PMLE[1:10,1:10])

	
	for i in range(totalWords+1):
		for j in range(21):
			if(int(classTotalWords[j])==0):PBE[i,j]=0
			else: PBE[i,j]=float(wordCount[i,j]+1)/(int(classTotalWords[j])+totalWords)			#(nk+1)/(n+|v|)
	
	#print("PBE")
	#print(PBE[1:10,1:10])		


def classifyNaiveBayesText():											#function to classify texts according to learned Model
	
	with open(testing_label) as csvfile4:								#reading testing_label csv file
		readCSV4 = csv.reader(csvfile4, delimiter=',')
		n=0
		for row in readCSV4:
			testClassData[n,1]=n+1										#Constructing array of docID vs ClassID
			testClassData[n,0]=int(row[0])																
			n+=1	
		csvfile4.close()
	
	#print(testClassData)
	
	testColumn_1=testClassData[:,0]										#Slicing only classIDs
	
	for x in range(20):
		L=[]
		testClassDocuments.append(L)									#constructing empty list of lists to contain class-wise DocIDs in test Data
		
		

	g=0	
	for x in list(testColumn_1):										#class-wise seperated lists of document names									
		if(x==1):testClassDocuments[0].append(g+1)
		elif(x==2):testClassDocuments[1].append(g+1)
		elif(x==3):testClassDocuments[2].append(g+1)
		elif(x==4):testClassDocuments[3].append(g+1)
		elif(x==5):testClassDocuments[4].append(g+1)
		elif(x==6):testClassDocuments[5].append(g+1)
		elif(x==7):testClassDocuments[6].append(g+1)
		elif(x==8):testClassDocuments[7].append(g+1)
		elif(x==9):testClassDocuments[8].append(g+1)
		elif(x==10):testClassDocuments[9].append(g+1)
		elif(x==11):testClassDocuments[10].append(g+1)
		elif(x==12):testClassDocuments[11].append(g+1)
		elif(x==13):testClassDocuments[12].append(g+1)
		elif(x==14):testClassDocuments[13].append(g+1)
		elif(x==15):testClassDocuments[14].append(g+1)
		elif(x==16):testClassDocuments[15].append(g+1)
		elif(x==17):testClassDocuments[16].append(g+1)
		elif(x==18):testClassDocuments[17].append(g+1)
		elif(x==19):testClassDocuments[18].append(g+1)
		elif(x==20):testClassDocuments[19].append(g+1)
		g+=1;
	
	
	with open(testing_data) as csvfile5:								#opening testing_data csv file
		readCSV5 = csv.reader(csvfile5, delimiter=',')
		m=0
		for row in readCSV5:
			testWordData[m,0]=int(row[0])								#array of docID vs wordID vs wordCount
			testWordData[m,1]=int(row[1])
			testWordData[m,2]=int(row[2])
			m+=1
				
		csvfile5.close()
	
	#print(testWordData)
	
	if(testOn=='0'):													#Test will be on training data 
		x=0;
		for L in range(11269):												
			max=-np.inf;
			maxClass=0;
			docID=int(classData[L,1])									#each Document is sampled
			#print(docID)
			for J in range(20):											#each document tested on each classID
				PWJ=math.log(classPriori[J])							#ln[P(Wj)]				
				#print(PWJ)
				PSum=float(0);
				i=x;
				#print(int(wordData[i,0]))
				while(int(wordData[i,0])==docID):
					#if(int(wordData[i,0])==docID):
					wordID=int(wordData[i,1])
					#print(wordID,J+1)
					PWIWJ=math.log(PBE[wordID,J+1])						#Summation(ln[P(Xi|Wj)]): For training data we use only Bayesian Estimators
					#print(PWIWJ)
					PSum+=PWIWJ
					#print(PSum)
					i+=1;
					#else: i+=1;
				PTotal=float(PWJ+PSum)
				#print("PT=%f"%PTotal)
				#print(J)
				if(PTotal>float(max)):									#classID (J) that maximizes the sum of two terms will be stored as predicted class
					max=PTotal
					maxClass=(J+1)
			x=i;
			predictedLabel[docID-1,0]=maxClass							#The argmax value is stored
			predictedLabel[docID-1,1]=docID
			#print(docID,maxClass)
		#print(predictedLabel)
		correctlyClassified=0
		for x in range(11269):
			if(classData[x,0]==predictedLabel[x,0]):
				correctlyClassified+=1;
				classwiseCorrectlyClassified[int(classData[x,0])-1]+=1;
				
		Accuracy=float(correctlyClassified)/11269
		S = 'Overall Accuracy= '+repr(Accuracy)+'  or  '+str(Accuracy*100)+' %'		#Printing Overall Accuracy
		print(S)
		print("Class Accuracy: ")
		for x in range(20):
			S = 'Class' + repr(x+1)+': ' + repr(float(classwiseCorrectlyClassified[x])/len(classDocuments[x]))
			print(S)
		
			
	elif(testOn=='1'):																#Similarly, prediction on testing data
		if(estimatorType=='0'):														#Using Bayesian Estimators
			x=0;
			for L in range(7505):												
				max=-np.inf;
				maxClass=0;
				docID=int(testClassData[L,1])
				#print(docID)
				for J in range(20):
					PWJ=math.log(classPriori[J])
					#print(PWJ)
					PSum=float(0);
					i=x;
					#print(int(wordData[i,0]))
					while(int(testWordData[i,0])==docID):
						#if(int(wordData[i,0])==docID):
						wordID=int(testWordData[i,1])
						#print(wordID,J+1)
						PWIWJ=math.log(PBE[wordID,J+1])
						#print(PWIWJ)
						PSum+=PWIWJ
						#print(PSum)
						i+=1;
						#else: i+=1;
					PTotal=float(PWJ+PSum)
					#print("PT=%f"%PTotal)
					#print(J)
					if(PTotal>float(max)):
						#print("LLOL")
						max=PTotal
						maxClass=(J+1)
				x=i;
				predictedLabel[docID-1,0]=maxClass										#The argmax value is stored
				predictedLabel[docID-1,1]=docID
				#print(docID,maxClass)
				#print(predictedLabel)
			correctlyClassified=0
			for x in range(7505):
				if(testClassData[x,0]==predictedLabel[x,0]):
					correctlyClassified+=1;
					classwiseCorrectlyClassified[int(testClassData[x,0])-1]+=1;
				
			Accuracy=float(correctlyClassified)/7505
			S = 'Overall Accuracy= '+repr(Accuracy)+'  or  '+str(Accuracy*100)+' %'		#Printing Overall Accuracy
			print(S)
			print("Class Accuracy: ")
		
			for x in range(20):
				S = 'Class' + repr(x+1)+': ' + repr(float(classwiseCorrectlyClassified[x])/len(testClassDocuments[x]))
				print(S)
			
		elif(estimatorType=='1'):
			x=0;
			for L in range(7505):												
				max=-np.inf;
				maxClass=0;
				docID=int(testClassData[L,1])
				#print(docID)
				for J in range(20):
					PWJ=math.log(classPriori[J])
					#print(PWJ)
					PSum=float(0);
					i=x;
					#print(int(wordData[i,0]))
					while(int(testWordData[i,0])==docID):
						#if(int(wordData[i,0])==docID):
						wordID=int(testWordData[i,1])
						#print(wordID,J+1)
						if(PMLE[wordID,J+1]!=0):PWIWJ=math.log(PMLE[wordID,J+1])
						else: PWIWJ=-1.00e+100
						#print(PWIWJ)
						PSum+=PWIWJ
						#print(PSum)
						i+=1;
						#else: i+=1;
					PTotal=float(PWJ+PSum)
					#print("PT=%f"%PTotal)
					#print(J)
					if(PTotal>float(max)):
						#print("LLOL")
						max=PTotal
						maxClass=(J+1)
				x=i;
				#print(docID,maxClass)
				predictedLabel[docID-1,0]=maxClass										#The argmax value is stored
				predictedLabel[docID-1,1]=docID
				#print(docID,maxClass)
				#print(predictedLabel)
			correctlyClassified=0
			for x in range(7505):
				if(testClassData[x,0]==predictedLabel[x,0]):
					correctlyClassified+=1;
					classwiseCorrectlyClassified[int(testClassData[x,0])-1]+=1;
				
			Accuracy=float(correctlyClassified)/7505
			S = 'Overall Accuracy= '+repr(Accuracy)+'  or  '+str(Accuracy*100)+' %'		#Printing Overall Accuracy
			print(S)
			print("Class Accuracy: ")
		
			for x in range(20):
				S = 'Class' + repr(x+1)+': ' + repr(float(classwiseCorrectlyClassified[x])/len(testClassDocuments[x]))	
				print(S)
		else:
			print("Argument error (estimatorType): Please revise argument")
	else:
		print("Argument error (dataType): Please revise argument")

		
		
confusionMatrix=np.zeros((21,21))		
def drawConfusionMatrix():																#Function to print Confusion Matrix 
	if(testOn=='0'):																	#For testing on training data
		for x in range(11269):
			docID=classData[x,1]
			expectedClass=int(classData[x,0])
			predictedClass=int(predictedLabel[x,0])
			confusionMatrix[expectedClass,predictedClass]+=1
		print("Confusion Matrix: ")
		print(confusionMatrix[1:,1:])
	
	if(testOn=='1'):																	#For testing on testing data 
		for x in range(7505):
			docID=testClassData[x,1]
			expectedClass=int(testClassData[x,0])
			predictedClass=int(predictedLabel[x,0])
			confusionMatrix[expectedClass,predictedClass]+=1
		print("Confusion Matrix: ")
		print(confusionMatrix[1:,1:])
			
	
learnNaiveBayesText()
classifyNaiveBayesText()
drawConfusionMatrix()
print("--- Execution Time %s (s) ---" % (time.time() - start_time))						#displays execution time 