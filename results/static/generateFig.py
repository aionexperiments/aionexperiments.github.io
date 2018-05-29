#!/usr/bin/python

import argparse
import glob, sys
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import plot, iplot
import plotly
from numpy import median
import matplotlib.pyplot as plt
plotly.offline.init_notebook_mode()

fig, ax = plt.subplots()
def draw_plot(data, edge_color, fill_color, outlier_prop):
    bp = ax.boxplot(data, 0, outlier_prop, patch_artist=True)
    for element in ['boxes', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)
    for patch in bp['boxes']:
        patch.set(facecolor=fill_color) 

def defineArguments():
    parser = argparse.ArgumentParser(prog="generateFig.py", description="Generates box plots from dictionaries.")
    parser.add_argument("-i", "--indir", help="The directory containing the dictionaries in \".data\" extension", required=True)
    parser.add_argument("-n", "--datasetname", help="The name of the dataset", required=True)
    parser.add_argument("-t", "--scoretype", help="The type of scores to consider", required=False, choices=["TRAIN", "TEST"], default="TRAIN")
    parser.add_argument("-r", "--metric", help="The metric to plot", required=False, default="F1 Score")# choices=["accuracy", "f1score", "precision", "recall", "specificity"])
    parser.add_argument("-f", "--figure", help="The type of the figure to generate", required=False, default="box", choices=["box", "box+line", "bar", "line"])
    parser.add_argument("-l", "--title", help="The title of the graph", required=False, default="Test")
    parser.add_argument("-w", "--width", help="The width of the graph", required=False, default=500, type=int)
    parser.add_argument("-e", "--height", help="The height of the graph", required=False, default=500, type=int)
    parser.add_argument("-g", "--legend", help="Whether to show a legend on the figure", required=False, default="yes", choices=["yes", "no"])
    parser.add_argument("-o", "--legendposition", help="The position of the legend", required=False, default="upper left")
    parser.add_argument("-p", "--plotter", help="The plotting library to use", required=False, default="matplotlib", choices=["matplotlib", "plotly"])
    return parser

def main():
    # Parse arguments
    argumentParser = defineArguments()
    arguments = argumentParser.parse_args()

    allfiles = glob.glob("%s/*.data" % arguments.indir)
    data = {}
    for f in allfiles:
        if f.lower().find(arguments.datasetname.lower()) == -1:
            print "[*] Skipping file \"%s\"" % f
            continue
        # Retrieve feature type
        if f.find("api") != -1:
            features = "api"
        elif f.find("basic") != -1:
            features = "basic"
        elif f.find("permission") != -1:
            features = "permission"
        else:
            features = "all"
        content = eval(open(f).read())
        for k in content[arguments.scoretype]:
            if k not in data.keys():
                data[k] = {}
            data[k][features] = content[arguments.scoretype][k]
 
    # Retrieve all classifiers
    #X = [l for l in data.keys()]
    #X.sort()
    X = ["KNN10", "KNN25", "KNN50", "KNN100", "KNN250", "KNN500", "SVM", "Trees10", "Trees25", "Trees50", "Trees75", "Trees100", "Ensemble"]
    Xs = ["K10", "K25", "K50", "K100", "K250", "K500", "SVM", "T10", "T25", "T50", "T75", "T100", "En"]
    #XS = X
    # c: cyan, m: magenta, y:yello, w, white
    # "v": triangle down, "8": octagon, "+": plus, "x": x, "D": diamond
    ####################
    # Box and whiskers #
    ####################
    if arguments.figure.find("box") != -1:
        if arguments.plotter == "matplotlib":
            # Build data
            d = {}
            metric = arguments.metric.lower().replace(' ', '')
            for x in X:
                if x not in d.keys():
                    d[x] = {"basic": [], "permission": [], "api": [], "all": []}
                
                d[x]["basic"] += data[x]["basic"][metric]
                d[x]["permission"] += data[x]["permission"][metric]
                d[x]["api"] += data[x]["api"][metric]
                d[x]["all"] += data[x]["all"][metric]

            yBasic = [d[learner]["basic"] for learner in d.keys()]
            yPerm = [d[learner]["permission"] for learner in d.keys()]
            yAPI = [d[learner]["api"] for learner in d.keys()]
            yAll = [d[learner]["all"] for learner in d.keys()]
            # Build plot
            x = range(1,len(X)+1)
            #plt.boxplot(yBasic, 0, 'ro') # Basic F1 score
            draw_plot(yBasic, 'red', 'pink', 'ro')
            draw_plot(yPerm, 'white', 'green', 'g^')
            draw_plot(yAPI, 'white', 'blue', 'bs')
            draw_plot(yAll, 'white', 'grey', 'kD')
            plt.xticks(x, Xs, rotation=45)
            #plt.plot(x, yPerm, 'g-^') # Perm F1 score
            #plt.plot(x, yAPI, 'b-s') # API F1 score
            #plt.plot(x, yAll, 'k-D') # All F1 score
            #plt.legend(['Basic %s' % arguments.metric, "Perm %s" % arguments.metric, "API %s" % arguments.metric, "All %s" % arguments.metric], loc=arguments.legendposition)
            plt.show()
            #plt.savefig('%s_static_%s_%s_line.pdf' % (arguments.datasetname, arguments.metric.replace(' ',''), arguments.scoretype))
            #plt.savefig('%s_static_%s_%s_line.pgf' % (arguments.datasetname, arguments.metric.replace(' ',''), arguments.scoretype))


        elif arguments.plotter == "plotly":
            # Build traces
            xBasic, xPerm, xAPI, xAll = [], [], [], []
            yBasic, yPerm, yAPI, yAll = [], [], [], []
            sBasic, sPerm, sAPI, sAll = [], [], [], []
            traces = []
            metric = arguments.metric.lower().replace(' ', '')
            for x in X:
                shortName = Xs[X.index(x)]
                xBasic += [shortName] * len(data[x]["basic"][metric])
                yBasic += data[x]["basic"][metric]
                sBasic += data[x]["basic"]["specificity"]
                xPerm += [shortName] * len(data[x]["permission"][metric])
                yPerm += data[x]["permission"][metric]
                sPerm += data[x]["permission"]["specificity"]
                xAPI += [shortName] * len(data[x]["api"][metric])
                yAPI += data[x]["api"][metric]
                sAPI += data[x]["api"]["specificity"]
                xAll += [shortName] * len(data[x]["all"][metric])
                yAll += data[x]["all"][metric]
                sAll += data[x]["all"]["specificity"]
            # Vertical boxes 
            traces.append(go.Box(y=yBasic, x=xBasic, name="Basic F1 score", marker=dict(color='#FF4136')))
            traces.append(go.Box(y=sBasic, x=xBasic, name="Basic Specificity", marker=dict(color='#f442b9')))
            traces.append(go.Box(y=yPerm, x=xPerm, name="Perm F1 Score", marker=dict(color='#3D9970')))
            traces.append(go.Box(y=sPerm, x=xPerm, name="Perm Specificity", marker=dict(color='#2aea53')))
            traces.append(go.Box(y=yAPI, x=xAPI, name="API F1 Score", marker=dict(color='#FF851B')))
            traces.append(go.Box(y=sAPI, x=xAPI, name="API Specificity", marker=dict(color='#edbe15')))
            traces.append(go.Box(y=yAll, x=xAll, name="All F1 Score", marker=dict(color='#6BAED6')))
            traces.append(go.Box(y=sAll, x=xAll, name="All Specificity", marker=dict(color='#15e5ed')))
            layout = go.Layout(xaxis=dict(showgrid=True, showline=True, showticklabels=True, gridwidth=2, tickangle=90, tickfont=dict(family='Old Standard TT, serif',size=14,color='black')), yaxis=dict(titlefont=dict(family='Old Standard TT, sans-serif', size=14, color='grey'),zeroline=True, tickfont=dict(family='Old Standard TT, serif',size=14,color='black')), boxmode='group', width=arguments.width, height=arguments.height)

            fig=go.Figure(data=traces, layout=layout)
            plot(fig, filename="%s.html" % arguments.title.lower().replace(' ', '_'))

    ##################
    # Scatter + Line #
    ##################
    elif arguments.figure == "line":
    # Build traces
        yBasic, yPerm, yAPI, yAll = [], [], [], []
        traces = []
        metric = arguments.metric.lower().replace(' ','')
        for x in X:
            yBasic.append(median(data[x]["basic"][metric]))
            yPerm.append(median(data[x]["permission"][metric]))
            yAPI.append(median(data[x]["api"][metric]))
            yAll.append(median(data[x]["all"][metric]))

        if arguments.plotter == "matplotlib":
            x = range(1,len(X)+1)
            plt.xticks(x, Xs, rotation=45)
            plt.plot(x, yBasic, 'r-o') # Basic F1 score
            plt.plot(x, yPerm, 'g-^') # Perm F1 score
            plt.plot(x, yAPI, 'b-s') # API F1 score
            plt.plot(x, yAll, 'k-D') # All F1 score
            # c: cyan, m: magenta, y:yello, w, white
            # "v": triangle down, "8": octagon, "+": plus, "x": x, "D": diamond
            plt.legend(['Basic %s' % arguments.metric, "Perm %s" % arguments.metric, "API %s" % arguments.metric, "All %s" % arguments.metric], loc=arguments.legendposition)
            #plt.show()
            plt.savefig('%s_static_%s_%s_line.pdf' % (arguments.datasetname, arguments.metric.replace(' ',''), arguments.scoretype))
            plt.savefig('%s_static_%s_%s_line.pgf' % (arguments.datasetname, arguments.metric.replace(' ',''), arguments.scoretype))

        elif arguments.plotter == "plotly":

            traces.append(go.Scatter(x=Xs, y=mBasic, mode="lines+markers", name="Basic F1 score", line=dict(shape='spline', width=2), marker=dict(symbol="circle", size=8, color='#FF4136')))
            traces.append(go.Scatter(x=Xs, y=sBasic, mode="lines+markers", name="Basic Specificity", line=dict(shape='spline', width=2), marker=dict(symbol="square", size=8, color='#f442b9')))
            traces.append(go.Scatter(x=Xs, y=mPerm, mode="lines+markers", name="Perm F1 score", line=dict(shape='spline', width=2), marker=dict(symbol="diamond", size=8, color='#3D9970')))
            traces.append(go.Scatter(x=Xs, y=sPerm, mode="lines+markers", name="Perm Specificity", line=dict(shape='spline', width=2), marker=dict(symbol="cross", size=8, color='#2aea53')))
            traces.append(go.Scatter(x=Xs, y=mAPI, mode="lines+markers", name="API F1 score", line=dict(shape='spline', width=2), marker=dict(symbol="x", size=8, color='#FF851B')))
            traces.append(go.Scatter(x=Xs, y=sAPI, mode="lines+markers", name="API Specificity", line=dict(shape='spline', width=2), marker=dict(symbol="triangle", size=8, color='#edbe15')))
            traces.append(go.Scatter(x=Xs, y=mAll, mode="lines+markers", name="All F1 score", line=dict(shape='spline', width=2), marker=dict(symbol="pentagon", size=8, color='#6BAED6')))
            traces.append(go.Scatter(x=Xs, y=sAll, mode="lines+markers", name="All Specificity", line=dict(shape='spline', width=2), marker=dict(symbol="star", size=8, color='#15e5ed')))
            layout = go.Layout(xaxis=dict(showgrid=True, showline=True, showticklabels=True, gridwidth=2, tickangle=90, tickfont=dict(family='Old Standard TT, serif',size=14,color='black')), yaxis=dict(titlefont=dict(family='Old Standard TT, sans-serif', size=14, color='grey'),zeroline=True, tickfont=dict(family='Old Standard TT, serif',size=16,color='black')), boxmode='group', width=arguments.width, height=arguments.height, legend=dict(font=dict(family='Old Standard TT, sans-serif',size=14, color='black')))

            fig=go.Figure(data=traces, layout=layout)
            plot(fig, filename="%s.html" % arguments.title.lower().replace(' ', '_'))



if __name__ == "__main__":
    main()
