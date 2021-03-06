# -*- coding: utf-8 -*-

import sys
from matplotlib import pyplot as plt

#ToDo: verify that my daily data candles match external sources
def plot_candles (ds = None, 
                  df = None, 
                  o = None,
                  h = None,
                  l = None,
                  c = None,
                  bShow = True):
    if ds is not None:
        if ds.f_df is not None:
            df = ds.f_df
        elif ds.df is not None:
            df = ds.df
        else:
            return
    if df is not None:
        o = df.Open
        h = df.High
        l = df.Low
        c = df.Close
    
    fig = plt.figure (figsize = (20, 10))
    plt.style.use ('ggplot')
    
    fn_plot_stick = lambda i: plt.plot ([i + 6, i + 6], [l[i], h[i]], color = ('blue' if c[i] > o[i] else 'red') )
    fn_plot_body = lambda i: plt.plot ([i + 6, i + 6], [o[i], c[i]], color = ('blue' if c[i] > o[i] else 'red'), linewidth=5)
    #for i in range (len (o)): 
    map (fn_plot_stick, range (len (o)))
    map (fn_plot_body, range (len (o)))
    #plt.plot ([i, i], [o, c], color = ('blue' if c[i] > o[i] else 'red') )
    
    if bShow:
        plt.grid (True)
        plt.show ()
    return fig

def plot_timeseries_vs_another (ts1 = None, ts2=None,
                   bSameAxis = False,
                   figsize=(10,5),
                   bMultiple = False,
                   bSave=False,
                   title = '',
                   label1 = '',
                   label2 = '',
                   y_bounds1 = None,
                   y_bounds2 = None,
                   color1 = 'red',
                   color2 = 'blue',
                   plot_filename=''):
    
    """
    Plots one timeseries vs another
    @params:
        ts1   - Optional  : 1st timeseries
        ts2   - Optional  : 2nd timeseries
        bSameAxis      - Optional  : if False, 2nd series uses vertical RHS axis
        figsize      - Optional  : size of the plot
        ...
    """
    if ts1 is None and ts2 is None:
        return plt.figure (figsize=figsize)

    if title == '':
        if label1 != '':
            title = label1
        else:
            title = 'Smtg vs Close'

    if not bMultiple:
        fig = plt.figure (figsize=figsize)

    plt.title (title)

    if bSameAxis:
        obj1 = obj2 = plt
    else:
        axes = plt.gca()
        ax = axes.twinx ()
        obj1 = axes
        obj2 = ax

    if ts1 is not None:
        obj1.plot(ts1, color='red', label=label1)
        if y_bounds1 is not None:
            obj1.set_ybound (lower=y_bounds1[0], upper=y_bounds1[1])

    if ts2 is not None:
        obj2.plot(ts2, color='blue', label=label2)
        if y_bounds2 is not None:
            obj2.set_ybound (lower=y_bounds2[0], upper=y_bounds2[1])

    if not bMultiple:
        plt.legend (loc='best')
        plt.show ()
        if bSave:
            fig.savefig(plot_filename)

    return fig


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 40, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    sys.stdout.flush()
    # Print New Line on Complete
    if iteration == total:
        print()