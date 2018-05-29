#!/usr/bin/python

import argparse
import glob, sys
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import plot, iplot
import plotly
from numpy import median
import numpy as np
from db import *
import matplotlib.pyplot as plt
plotly.offline.init_notebook_mode()

all_dp_query = "SELECT dpIteration,learnerName,dpFeature,dpType,dpAccuracy,dpFScore,dpPrecision,dpRecall,dpSpecificity FROM datapoint,learner WHERE dpLearner=learnerID AND learnerName='%s' AND dpType='%s';"
learners = ["KNN10", "KNN25", "KNN50", "KNN100", "KNN250", "KNN500", "SVM", "Trees10", "Trees25", "Trees50", "Trees75", "Trees100", "Ensemble"]
learners_short = ["K10", "K25", "K50", "K100", "K250", "K500", "SVM", "T10", "T25", "T50", "T75", "T100", "En"]

def defineArguments():
    parser = argparse.ArgumentParser(prog="generateFig.py", description="Generates box plots from dictionaries.")
    parser.add_argument("-d", "--database", help="The database containing the saved results", required=True)
    parser.add_argument("-n", "--datasetname", help="The name of the dataset", required=True)
    parser.add_argument("-t", "--scoretype", help="The score type to consider", required=False, default="TEST", choices=["TRAIN", "TEST"])
    parser.add_argument("-r", "--learner", help="The name of the classifier to focus on", required=True)
    parser.add_argument("-u", "--feature", help="The feature type to plot", required=False, choices=["dynamic", "hybrid"])
    parser.add_argument("-f", "--figure", help="The type of the figure to generate", required=False, default="box", choices=["box", "box+line", "line"])
    parser.add_argument("-w", "--width", help="The width of the graph", required=False, default=500, type=int)
    parser.add_argument("-e", "--height", help="The height of the graph", required=False, default=500, type=int)
    parser.add_argument("-p", "--plotter", help="The library to use for plotting figures", required=False, default="matplotlib", choices=["matplotlib", "plotly"])
    parser.add_argument("-o", "--legendposition", help="The position of the matplotlib legend", required=False, default="upper left")
    parser.add_argument("-c", "--color", help="The color scale of some matplotlib figures", required=False, default="rgb", choices=["rgb", "cmyk"])
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
    c = database.execute(all_dp_query % (arguments.learner, arguments.scoretype))
    datapoints = c.fetchall()
    if len(datapoints) < 1:
        print "[*] Could not retrieve rows using the query above. Exiting"
        return False

    iterations = []
    for dp in datapoints:
        if int(dp[0]) <= 10:
            iteration = "itn.%s" % dp[0]
        if iteration not in iterations:
            iterations.append(iteration)

        feature = dp[2]
        score = dp[3]

        if iteration not in data.keys():
            data[iteration] = {}
        if feature not in data[iteration].keys():
            data[iteration][feature] = {"accuracy": [], "f1score": [], "precision": [], "recall": [], "specificity": []}

        data[iteration][feature]["accuracy"].append(dp[4])
        data[iteration][feature]["f1score"].append(dp[5])
        data[iteration][feature]["precision"].append(dp[6])
        data[iteration][feature]["recall"].append(dp[7])
        data[iteration][feature]["specificity"].append(dp[8])


    CMYK = ["#c0c0c0", "#808080", "#505050", "#000000"] # Grayscale colors
    RGB = ["#ff4136", "#3d9970", "#ff851b", "#6baed6", "#808389", "48494c"] # Normal colors
    ####################
    # Box and whiskers #
    ####################
    #iterations.sort() # Sort iterations
    if arguments.figure.find("box") != -1:
        if arguments.plotter == "plotly":
            # Include the best achieving static set on Piggybacking (permission)
            content = eval(open("piggybacking_static_permission_dict.data").read())
            fStatic_Perm = median(content[arguments.scoretype][arguments.learner]["f1score"])
            sStatic_Perm = median(content[arguments.scoretype][arguments.learner]["specificity"])
            # Build traces
            xF1, yF1, mF1 = [], [], []
            xSp, ySp, mSp = [], [], []
            xF1_hybrid, yF1_hybrid, mF1_hybrid = [], [], []
            xSp_hybrid, ySp_hybrid, mSp_hybrid = [], [], []
            xF1_static, yF1_static, mF1_static = [], [], []
            xSp_static, ySp_static, mSp_static = [], [], []
            traces = []
            for i in iterations:
                # F1 + spec. dynamic
                xF1 += [i] * len(data[i]["dynamic"]["f1score"])
                yF1 += data[i]["dynamic"]["f1score"]
                mF1.append(median(data[i]["dynamic"]["f1score"]))
                xSp += [i] * len(data[i]["dynamic"]["specificity"])
                ySp += data[i]["dynamic"]["specificity"]
                mSp.append(median(data[i]["dynamic"]["specificity"]))
                # F1 + spec. hybrid
                xF1_hybrid += [i] * len(data[i]["hybrid"]["f1score"])
                yF1_hybrid += data[i]["hybrid"]["f1score"]
                mF1_hybrid.append(median(data[i]["hybrid"]["f1score"]))
                xSp_hybrid += [i] * len(data[i]["hybrid"]["specificity"])
                ySp_hybrid += data[i]["hybrid"]["specificity"]
                mSp_hybrid.append(median(data[i]["hybrid"]["specificity"]))

            # Vertical boxes
            traces.append(go.Box(y=yF1, x=xF1, name="Dynamic F1 Score", marker=dict(color='#FF4136')))
            traces.append(go.Box(y=ySp, x=xSp, name="Dynamic Specificity", marker=dict(color='#3D9970')))
            traces.append(go.Box(y=yF1_hybrid, x=xF1_hybrid, name="Hybrid F1 Score", marker=dict(color='#FF851B')))
            traces.append(go.Box(y=ySp_hybrid, x=xSp_hybrid, name="Hybrid Specificity", marker=dict(color='#6BAED6')))
            traces.append(go.Scatter(x=iterations, y=[fStatic_Perm]*len(mF1), mode="lines", name="Static F1 Score", marker=dict(color='#808389')))
            traces.append(go.Scatter(x=iterations, y=[sStatic_Perm]*len(mSp), mode="lines", name="Static Specificity", marker=dict(color='#48494c')))
            if arguments.figure == "box+line":
                traces.append(go.Scatter(x=xF1, y=mF1, name="", mode="lines", marker=dict(color='#FF4136')))
                traces.append(go.Scatter(x=xSp, y=mSp, name="", mode="lines", marker=dict(color='#3D9970')))
                traces.append(go.Scatter(x=xF1_test, y=mF1_test, name="", mode="lines", marker=dict(color='#FF851B')))
                traces.append(go.Scatter(x=xSp_test, y=mSp_test, name="", mode="lines", marker=dict(color='#6BAED6')))

            layout = go.Layout(xaxis=dict(showgrid=True, showline=True, showticklabels=True, gridwidth=2, tickangle=90, tickfont=dict(family='Old Standard TT, serif',size=12,color='black')), yaxis=dict(titlefont=dict(family='Old Standard TT, sans-serif', size=14, color='grey'),zeroline=True, tickfont=dict(family='Old Standard TT, serif',size=12,color='black')), boxmode='group', width=arguments.width, height=arguments.height, legend=dict(orientation="h", xanchor="center", y=1.1, x=0.5))
  
            fig=go.Figure(data=traces, layout=layout)
            plot(fig, filename="%s_active_%s_%s_box.html" % (arguments.datasetname, arguments.scoretype, arguments.learner))
    
    ##################
    # Scatter + Line #
    ##################
    elif arguments.figure == "line":
        # Include the best achieving static set on Piggybacking (permission)
        content = eval(open("piggybacking_static_permission_dict.data").read())
        fStatic_Perm = median(content[arguments.scoretype][arguments.learner]["f1score"])
        sStatic_Perm = median(content[arguments.scoretype][arguments.learner]["specificity"])

        if arguments.plotter == "matplotlib":
            colors = RGB if arguments.color == "rgb" else CMYK
            # Build traces
            mF1, mSp, mF1_hybrid, mSp_hybrid = [], [], [], []
            traces = []
            for i in iterations:
                mF1.append(median(data[i]["dynamic"]["f1score"]))
                mSp.append(median(data[i]["dynamic"]["specificity"]))
                mF1_hybrid.append(median(data[i]["hybrid"]["f1score"]))
                mSp_hybrid.append(median(data[i]["hybrid"]["specificity"]))

            x = range(1,len(iterations)+1)
            plt.xticks(x, iterations, rotation=45)
            plt.plot(x, mF1, color=colors[0], marker='o') # Dynamic F1 score
            plt.plot(x, mSp, color=colors[1], marker='^') # Dynamic specificity score
            plt.plot(x, mF1_hybrid, color=colors[2], marker='s') # Hybrid F1 score
            plt.plot(x, mSp_hybrid, color=colors[3], marker='D') # Hybrid specificity score
            if arguments.color == "rgb":
                plt.plot(x, [fStatic_Perm]*len(mF1), 'c--', linewidth=2) # F1 Permissions (Static)
                plt.plot(x, [sStatic_Perm]*len(mSp), 'm--', linewidth=2) # Specificity Permissions (Static)
            else:
                plt.plot(x, [fStatic_Perm]*len(mF1), color=colors[0], linestyle='dashed', linewidth=2) # F1 Permissions (Static)
                plt.plot(x, [sStatic_Perm]*len(mSp), color=colors[3], linestyle='dashed', linewidth=2) # Specificity Permissions (Static)

            plt.legend(["Dynamic F1 Score", "Dynamic Specificty", "Hybrid F1 Score", "Hybrid Specificity", "Static F1 Score", "Static Specificity" ], loc=arguments.legendposition, fontsize="small").get_frame().set_alpha(0.5)
            #plt.show()
            plt.savefig("%s_active_f1spec_%s_%s_line.pdf" % (arguments.datasetname, arguments.learner, arguments.scoretype))
            plt.savefig("%s_active_f1spec_%s_%s_line.pgf" % (arguments.datasetname, arguments.learner, arguments.scoretype))
            
        elif arguments.plotter == "plotly":
            # Build traces
            mF1, mSp, mF1_hybrid, mSp_hybrid = [], [], [], []
            traces = []
            for i in iterations:
                mF1.append(median(data[i]["dynamic"]["f1score"]))
                mSp.append(median(data[i]["dynamic"]["specificity"]))
                mF1_hybrid.append(median(data[i]["hybrid"]["f1score"]))
                mSp_hybrid.append(median(data[i]["hybrid"]["specificity"]))


            traces.append(go.Scatter(x=iterations, y=mF1, mode="markers+lines", name="Dynamic F1 Score", marker=dict(color='#FF4136', symbol="circle", size=10)))
            traces.append(go.Scatter(x=iterations, y=mSp, mode="markers+lines", name="Dynamic Specificity", marker=dict(color='#3D9970', symbol="triangle-up", size=10)))
            traces.append(go.Scatter(x=iterations, y=mF1_hybrid, mode="markers+lines", name="Hybrid F1 score", marker=dict(color='#FF851B', symbol="square", size=10)))
            traces.append(go.Scatter(x=iterations, y=mSp_hybrid, mode="markers+lines", name="Hybrid Specificity", marker=dict(color='#6BAED6', symbol="diamond", size=10)))
            traces.append(go.Scatter(x=iterations, y=[fStatic_Perm]*len(mF1), mode="lines", name="Static F1 Score", marker=dict(color='#808389', size=10)))
            traces.append(go.Scatter(x=iterations, y=[sStatic_Perm]*len(mSp), mode="lines", name="Static Specificity", marker=dict(color='#48494c', size=10)))

            layout = go.Layout(xaxis=dict(showgrid=True, showline=True, showticklabels=True, gridwidth=2, tickangle=90, tickfont=dict(family='Old Standard TT, serif',size=12,color='black')), yaxis=dict(titlefont=dict(family='Old Standard TT, sans-serif', size=14, color='grey'),zeroline=True, tickfont=dict(family='Old Standard TT, serif',size=12,color='black')), boxmode='group', width=arguments.width, height=arguments.height, legend=dict(orientation="h", xanchor="center", y=1.1, x=0.5))

            fig=go.Figure(data=traces, layout=layout)
            plot(fig, filename='%s_active_%s_%s_line.html' % (arguments.datasetname, arguments.scoretype, arguments.learner))





if __name__ == "__main__":
    main()
