import pandas as pd
from pathlib import Path
import sys
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# Data loading
data = pd.read_csv('/home/jany/PycharmProjects/ML_App/data/titanic/train.csv')
print(data.head())

fig = go.Figure(data=[go.Table(header=dict(values=data.columns),
                 cells=dict(values=[data[i] for i in data.columns]))
                     ])
#fig.show()

# We can see, on the following plot, that most of the people died, only 38% of the passenger survived.
fig = px.pie(data, names='Survived', title='Passenger Survival')
#fig.show()

# We see that most of the passergers embarked from Southampton.
fig = go.Figure(data=[go.Pie(labels=data['Embarked'], pull=[.1, .15, .15, 0])])
#fig.show()

# On the following subplot, we observed the surviving rate of each place of onbording. People from Southampton died more often than the other.
# So there is a correlation between the place of unbording and the probaility to die.
fig = make_subplots(rows=1, cols=3, specs=[[{"type": "pie"}, {"type": "pie"}, {"type": "pie"}]])
fig.add_trace(
            go.Pie(labels=data.loc[data['Embarked'] == 'C']['Survived'], pull = [.1, .1],
                   title = 'Embarked C vs. Survived'), row=1, col=1)

fig.add_trace(
            go.Pie(labels=data.loc[data['Embarked'] == 'S']['Survived'], pull = [.07, .07],
                   title = 'Embarked S vs. Survived'),row=1, col=2)

fig.add_trace(
            go.Pie(labels=data.loc[data['Embarked'] == 'Q']['Survived'], pull = [.1, .1],
                   title = 'Embarked Q vs. Survived'), row=1, col=3)


fig.update_layout(height=500, width=800, title_text="Gene Expression Features")
#fig.show()

# We can see here the repartition of the passengers age. Most of the prassengers are under 40 yo.
fig = px.histogram(data, x='Age', nbins=50, histnorm='probability density')
#fig.show()

# We can see here the repartition of the passengers class against their age. Older people tend to be in 1st, and youger in2nd and 3rd.
fig = px.box(data, x='Pclass', y="Age", points="all")
#fig.show()


# So now we want to look at thje surviving rate against the passenger class. We see that 3ed class passenger died 25% more than the second
# and 38% more than the firt. Moreover 2nd died 15% more than the first. So we have a huge correlation between passenger's class and survivig rate.
# By extention we do have a correlation between passenger's age and surviving.
fig = make_subplots(rows=1, cols=3, specs=[[{"type": "pie"}, {"type": "pie"}, {"type": "pie"}]])
fig.add_trace(
            go.Pie(labels=data.loc[data['Pclass'] == 1]['Survived'], pull = [.1, .1],
                   title = 'Pclass 1 vs. Survived'), row=1, col=1)

fig.add_trace(
            go.Pie(labels=data.loc[data['Pclass'] == 2]['Survived'], pull = [.07, .07],
                   title = 'Pclass 2 vs. Survived'),row=1, col=2)

fig.add_trace(
            go.Pie(labels=data.loc[data['Pclass'] == 3]['Survived'], pull = [.1, .1],
                   title = 'Pclass 3 vs. Survived'), row=1, col=3)
#fig.show()

# Let's do the same with the Sex of the passenger. We see that men represent 65% of the poplation and tha women almost
# survive 4 times more that men does, so there is a big correlation here.
fig = px.pie(data, names='Sex', title='Passenger sex')
#fig.show()


fig = make_subplots(rows=1, cols=3, specs=[[{"type": "pie"}, {"type": "pie"}, {"type": "pie"}]])
fig.add_trace(
            go.Pie(labels=data.loc[data['Sex'] == 'male']['Survived'], pull = [.1, .1],
                   title = 'Male vs. Survived'), row=1, col=1)

fig.add_trace(
            go.Pie(labels=data.loc[data['Sex'] == 'female']['Survived'], pull = [.07, .07],
                   title = 'Female vs. Survived'),row=1, col=2)
#fig.show()

# Let's check if there is any correlation between the meaningfull features we just find.

# First, between the onbording spot and the passenger's class. We see that onbording S and 3rd class are correlated.
# We have to remember that passsenger from Southampton are most likely to die that the other and so are passenger from
# 3rd class. So this correlation is important because they seems to be overlaping each other in the explanation of the
# taget variable.
fig = px.density_heatmap(data, x="Embarked", y="Pclass",
                        height=500, width=500)
#fig.show()


# Finaly the correlation between siblings, spouses, parents, children and surviving. The more the family is develop
# the more yon tend to die on titanic.
fig = px.density_contour(data, x="SibSp", y="Parch", color='Survived',
                        height=400, width=800)
fig.update_traces(contours_coloring="fill", contours_showlabels = True)
fig.show()