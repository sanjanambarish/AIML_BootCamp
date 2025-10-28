#importing dependency
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 


#import dataset 
#x = number of hours studied
#y = pass(1) or fail (0)
x= np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
y= np.array([[0],[0],[0],[0],[1],[1],[1],[1],[1],[1]])

#split the data into train and test 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#train
model=LogisticRegression()
model.fit(x_train,y_train)

#test
y_pred=model.predict(x_test)

print(f"actual labels:{y_test}")
print(f"predicted labels:{y_pred}")

#accuracy 
print(f"accuracy score:{accuracy_score(y_test,y_pred)}")

#make prediction on unseen data
hours=np.array([[4.5],[3],[50],[1]])
result=model.predict(hours)
for h,r in zip(hours,result):
 print(f"if u study for {h} hours :{" pass " if r else "fail"}")


