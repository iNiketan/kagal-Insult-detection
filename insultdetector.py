import insult_analysis
from insult_analysis import Preprocess


test_input = ["That you are an idiot who understands neither taxation nor women's health."]
ip = Preprocess()
cleared_s = ip.cleaning(test_input[0])
print(cleared_s)
cleared = [cleared_s]
bow = insult_analysis.bagofwords(cleared)
print(bow)
pred = insult_analysis.naive_clf.predict(bow)
print(pred)
if pred == 1:
    print('insult')
else:
    print('not insult')



