import matplotlib.pyplot as plt
import seaborn as sns
from numpy.core.fromnumeric import mean
from numpy.lib.function_base import median
import pandas as pd

data = pd.read_csv('collected-data-annotation - Copy.csv')

# print(data.head())
file = open('eda-result.txt', 'w')
# print('total data points: {}'.format(sum(data['count']), file=file))
# print(
#     "total data points:{}".format(sum(data['count'])), 
#     file=file
#     )
# print(
#     "Mean age:{}".format(mean(data['age'])), 
#     file=file
#     )
# print(
#     "Median age:{}".format(median(data['age'])), 
#     file=file
#     )
# print(
#     "Mean age for males:{}".format(median(data[data['gender'] == -1]['age'])), 
#     file=file
#     )
# print(
#     "Mean age for females:{}".format(median(data[data['gender'] == 1]['age'])), 
#     file=file
#     )
# print(
#     "Outlier age count:{}".format(sum((data['age'] > 40))), 
#     file=file
#     )

# print(
#     "Outlier age count males:{}".format(sum((data[data['gender'] == -1]['age'] > 40))), 
#     file=file
#     )
# print(
#     "Outlier age count females:{}".format(sum((data[data['gender'] == 1]['age'] > 40))), 
#     file=file
#     )
genderCount = [ 
    sum(data[data['gender'] == -1]['count']),
    sum(data[data['gender'] == 1]['count']),
    ]
# print(genderCount)
# patches, texts = plt.pie(genderCount)
# plt.legend(patches, ['male', 'female'])
# sns.displot(data=data, x='age', hue='gender')
sns.displot(data=data, x='age', hue='gender', kde=True)
# plt.bar(x=genderCount)
#  kind='kde')
# plt.axis('equal')
plt.show()