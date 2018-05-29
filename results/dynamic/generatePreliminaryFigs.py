#!/usr/bin/python

import argparse
import glob, sys
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import plot, iplot
import plotly
import numpy as np
from numpy import median
from db import *
import matplotlib.pyplot as plt
plotly.offline.init_notebook_mode()

all_dp_query = "SELECT dpIteration,learnerName,dpFeature,dpType,dpAccuracy,dpFScore,dpPrecision,dpRecall,dpSpecificity FROM datapoint,learner WHERE dpLearner=learnerID AND dpIteration='1' AND dpType='%s'"
learners = ["KNN10", "KNN25", "KNN50", "KNN100", "KNN250", "KNN500", "SVM", "Trees10", "Trees25", "Trees50", "Trees75", "Trees100", "Ensemble"]
learners_short = ["K10", "K25", "K50", "K100", "K250", "K500", "SVM", "T10", "T25", "T50", "T75", "T100", "En"]

def defineArguments():
    parser = argparse.ArgumentParser(prog="generateFig.py", description="Generates box plots from dictionaries.")
    parser.add_argument("-d", "--database", help="The database containing the saved results", required=True)
    parser.add_argument("-n", "--datasetname", help="The name of the dataset", required=True)
    parser.add_argument("-s", "--scoretype", help="The type of score to consider", required=False, default="TEST", choices=["TRAIN", "TEST"])
    parser.add_argument("-m", "--metric", help="The type of metric to plot", required=False, default="F1 Score")
    parser.add_argument("-f", "--figure", help="The type of the figure to generate", required=False, default="box", choices=["box", "box+line", "scatter", "bar"])
    parser.add_argument("-w", "--width", help="The width of the graph", required=False, default=500, type=int)
    parser.add_argument("-e", "--height", help="The height of the graph", required=False, default=500, type=int)
    parser.add_argument("-p", "--plotter", help="The library to use for plotting", required=False, default="matplotlib", choices=["matplotlib", "plotly"])
    parser.add_argument("-o", "--legendposition", help="The position of the matplotlib legend", required=False, default="upper left")
    parser.add_argument("-c", "--colors", help="The color scale of figures", required=False, default="rgb", choices=["rgb", "cmyk"])
    return parser

def main():
    # Parse arguments
    argumentParser = defineArguments()
    arguments = argumentParser.parse_args()

    database = AionDB(arguments.database)
    if not database:
        print "[*] Could not establish a connection with the database \"%s\"" % arguments.database
        return False

    
    data = {}
    # Retrieve datapoints
    c = database.execute(all_dp_query % (arguments.scoretype))
    datapoints = c.fetchall()
    if len(datapoints) < 1:
        print "[*] Could not retrieve rows using the query above. Exiting"
        return False
    for dp in datapoints:
        learner = dp[1]
        feature = dp[2]
        score = dp[3]
 
        if learner not in data.keys():
            data[learner] = {}
        if feature not in data[learner].keys():
            data[learner][feature] = {"accuracy": [], "f1score": [], "precision": [], "recall": [], "specificity": []}

        data[learner][feature]["accuracy"].append(dp[4])
        data[learner][feature]["f1score"].append(dp[5])
        data[learner][feature]["precision"].append(dp[6])
        data[learner][feature]["recall"].append(dp[7])
        data[learner][feature]["specificity"].append(dp[8])     
 
    CMYK = ["#c0c0c0", "#808080", "#505050", "#000000"] # Grayscale colors
    RGB = ["#ff4136", "#3d9970", "#ff851b", "#6baed6", "#808389", "48494c"] # Normal colors
    ####################
    # Box and whiskers #
    ####################
    if arguments.figure.find("box") != -1:
        if arguments.plotter == "matplotlib":
            print "[*] Not implemented for \"matplotlib\""
            return True 
        elif arguments.plotter == "plotly":
            # Include the best achieving static set on Piggybacking (permission)
            content = eval(open("piggybacking_static_permission_dict.data").read())
            mF1_static, mSp_static = [], []
            # Build traces
            xF1, yF1, mF1 = [], [], []
            xSp, ySp, mSp = [], [], []
            xF1_hybrid, yF1_hybrid, mF1_hybrid = [], [], []
            xSp_hybrid, ySp_hybrid, mSp_hybrid = [], [] ,[] 
            xF1_static, yF1_static, mF1_static = [], [], []
            xSp_static, ySp_static, mSp_static = [], [] ,[] 
            traces = []
            for l in learners:
                shortName = learners_short[learners.index(l)]
                # Dynamic
                xF1 += [shortName] * len(data[l]["dynamic"]["f1score"])
                yF1 += data[l]["dynamic"]["f1score"]
                mF1.append(median(data[l]["dynamic"]["f1score"]))
                xSp += [shortName] * len(data[l]["dynamic"]["specificity"])
                ySp += data[l]["dynamic"]["specificity"]
                mSp.append(median(data[l]["dynamic"]["specificity"]))
                # Hybrid
                xF1_hybrid += [shortName] * len(data[l]["hybrid"]["f1score"])
                yF1_hybrid += data[l]["hybrid"]["f1score"]
                mF1_hybrid.append(median(data[l]["hybrid"]["f1score"]))
                xSp_hybrid += [shortName] * len(data[l]["hybrid"]["specificity"])
                ySp_hybrid += data[l]["hybrid"]["specificity"]
                mSp_hybrid.append(median(data[l]["hybrid"]["specificity"]))
                # Static (permission)
                xF1_static += [shortName] * len(content["TEST"][l]["f1score"])
                yF1_static += content["TEST"][l]["f1score"]
                mF1_static.append(median(content["TEST"][l]["f1score"]))
                xSp_static += [shortName] * len(content["TEST"][l]["specificity"])
                ySp_static += content["TEST"][l]["specificity"]
                mSp_static.append(median(content["TEST"][l]["specificity"]))

            # Vertical boxes
            traces.append(go.Box(y=yF1, x=xF1, name="F1 score (dynamic)", marker=dict(color='#FF4136')))
            traces.append(go.Box(y=ySp, x=xSp, name="Specificity (dynamic)", marker=dict(color='#3D9970')))
            traces.append(go.Box(y=yF1_hybrid, x=xF1_hybrid, name="F1 score (hybrid)", marker=dict(color='#FF851B')))
            traces.append(go.Box(y=ySp_hybrid, x=xSp_hybrid, name="Specificity (hybrid)", marker=dict(color='#6BAED6')))
            traces.append(go.Box(x=xF1_static, y=yF1_static, name="F1 Score (static)",  marker=dict(color='#808389')))
            traces.append(go.Box(x=xSp_static, y=ySp_static, name="Specificity (static)", marker=dict(color='#48494c')))
            if arguments.figure == "box+line":
                traces.append(go.Scatter(x=xF1, y=mF1, name="", mode="lines", line=dict(shape='spline'), marker=dict(color='#FF4136')))
                traces.append(go.Scatter(x=xSp, y=mSp, name="", mode="lines", line=dict(shape='spline'), marker=dict(color='#3D9970')))
                traces.append(go.Scatter(x=xF1_hybrid, y=mF1_hybrid, name="", mode="lines", line=dict(shape='spline'), marker=dict(color='#FF851B')))
                traces.append(go.Scatter(x=xSp_hybrid, y=mSp_hybrid, name="", mode="lines", line=dict(shape='spline'), marker=dict(color='#6BAED6')))

            layout = go.Layout(xaxis=dict(showgrid=True, showline=True, showticklabels=True, gridwidth=2, tickangle=90, tickfont=dict(family='Old Standard TT, serif',size=14,color='black')), yaxis=dict(titlefont=dict(family='Old Standard TT, sans-serif', size=14, color='grey'),zeroline=True, tickfont=dict(family='Old Standard TT, serif',size=14,color='black')), boxmode='group', width=arguments.width, height=arguments.height, legend=dict(orientation="h", xanchor="center", y=1.1, x=0.5))

            fig=go.Figure(data=traces, layout=layout)
            plot(fig, filename="%s_preliminary_f1spec_%s_box.html" % (arguments.datasetname, arguments.scoretype))

    #######
    # Bar #
    #######
    elif arguments.figure == "bar":
        # Include the best achieving static set on Piggybacking (permission)
        content = eval(open("piggybacking_static_permission_dict.data").read())
        mF1, mSp, mF1_hybrid, mSp_hybrid = [], [], [], []
        mF1_static, mSp_static = [], []
        traces = []
        for l in learners:
            mF1.append(median(data[l]["dynamic"]["f1score"]))
            mSp.append(median(data[l]["dynamic"]["specificity"]))
            mF1_hybrid.append(median(data[l]["hybrid"]["f1score"]))
            mSp_hybrid.append(median(data[l]["hybrid"]["specificity"]))
            mF1_static.append(median(content["TEST"][l]["f1score"]))
            mSp_static.append(median(content["TEST"][l]["specificity"]))

        if arguments.plotter == "matplotlib":
            x = np.arange(1, len(learners)+1)
            width = 0.25
            fig, ax = plt.subplots()
            colors = RGB if arguments.colors == "rgb" else CMYK
            rectsF1Dynamic = ax.bar(x, mF1, width, color=colors[0])
            #rectsSpDynamic = ax.bar(x, mSp, width, color=colors[0])
            rectsF1Hybrid = ax.bar(x+width, mF1_hybrid, width, color=colors[1])
            #rectsSpHybrid = ax.bar(x+width, mSp_hybrid, width, color=colors[1])
            rectsF1Static = ax.bar(x+(2*width), mF1_static, width, color=colors[2])
            #rectsSpStatic = ax.bar(x+(2*width), mSp_static, width, color=colors[2])
            
            ax.set_xticks(x)
            plt.xticks(rotation=45)
            ax.set_xticklabels(learners_short)
            ax.legend((rectsF1Dynamic, rectsF1Hybrid, rectsF1Static), ('Dynamic F1 Score', 'Hybrid F1 Score', 'Static F1 Score'), loc=arguments.legendposition)
            #ax.legend((rectsSpDynamic, rectsSpHybrid, rectsSpStatic), ('Dynamic Specificity', 'Hybrid Specificity', 'Static Specificity'), loc=arguments.legendposition)
            #plt.show()
            plt.savefig('%s_dynamic_%s_bar.pdf' % (arguments.datasetname, arguments.scoretype), width=20, height=15)
            plt.savefig('%s_dynamic_%s_bar.pgf' % (arguments.datasetname, arguments.scoretype), width=20, height=15)

    ###########
    # Scatter #
    ###########
    elif arguments.figure == "scatter":
        # Include the best achieving static set on Piggybacking (permission)
        content = eval(open("piggybacking_static_permission_dict.data").read())
        m, m_hybrid, m_static = [], [], []
        traces = []
        metric = arguments.metric.lower().replace(' ', '')
        colors = RGB if arguments.colors == "rgb" else CMYK
        for l in learners:
            m.append(median(data[l]["dynamic"][metric]))
            m_hybrid.append(median(data[l]["hybrid"][metric]))
            m_static.append(median(content["TEST"][l][metric]))
        if arguments.plotter == "matplotlib":
            # Build traces
            x = range(1,len(learners)+1)
            plt.xticks(x, learners_short, rotation=45)
            plt.plot(x, m, color=colors[0], marker='o', linestyle='') # Dynamic F1 score
            plt.plot(x, m_hybrid, color=colors[1], marker='s', linestyle='') # Hybrid F1 score
            plt.plot(x, m_static, color=colors[2], marker='^', linestyle='') # F1 Permissions (Static)
            plt.legend(["Dynamic %s" % arguments.metric, "Hybrid %s" % arguments.metric, "Static %s" % arguments.metric], loc=arguments.legendposition)
            #plt.show()
            plt.savefig('%s_dynamic_%s_%s_scatter.pdf' % (arguments.datasetname, arguments.scoretype, arguments.metric))
            plt.savefig('%s_dynamic_%s_%s_scatter.pgf' % (arguments.datasetname, arguments.scoretype, arguments.metric))

        elif arguments.plotter == "plotly":
            traces.append(go.Scatter(x=learners, y=m, mode="markers", name="Dynamic %s" % arguments.metric, marker=dict(color=colors[0], symbol="circle", size=8)))
            traces.append(go.Scatter(x=learners, y=m_hybrid, mode="markers", name="Hybrid %s" % arguments.metric, marker=dict(color=colors[1], symbol="square", size=8)))
            traces.append(go.Scatter(x=learners, y=m_static, mode="markers", name="Static %s" % arguments.metric, marker=dict(color=colors[2], symbol="triangle-up", size=8)))

            layout = go.Layout(xaxis=dict(showgrid=True, showline=True, showticklabels=True, gridwidth=2, tickangle=90, tickfont=dict(family='Old Standard TT, serif',size=14, color='black')), yaxis=dict(titlefont=dict(family='Old Standard TT, sans-serif', size=14, color='grey'),zeroline=True, tickfont=dict(family='Old Standard TT, serif',size=14,color='black')), boxmode='group', width=arguments.width, height=arguments.height)#, legend=dict(orientation="h", xanchor="center", y=1.1, x=0.5))

            fig=go.Figure(data=traces, layout=layout)
            plot(fig, filename="%s_preliminary_%s_%s_scatter.html" % (arguments.datasetname, arguments.scoretype, arguments.metric))

if __name__ == "__main__":
    main()
