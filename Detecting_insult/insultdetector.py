import bowmodel
from Preprocess import Preprocess as Ipreprocess

print("please input comment")
s = input()
ip = Ipreprocess()
cleared_s = ip.cleaning(s)

bow = bowmodel.bagofwords(cleared_s)
pred = bowmodel.naive_clf.predict(bow)

if pred == 1:
    print('insult')
else:
    print('not insult')

