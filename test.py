import sklearn as sklearn
import numpy
print(numpy.__path__)
print(sklearn.__path__)

a = numpy.array([1,2,3])
b = numpy.array([2,4,6])

print(sklearn.metrics.explained_variance_score(a, b))
