
def plot_metrics(metrics, filename):
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    matplotlib.use('agg')

    fig, ax = plt.subplots()
    #with plt.style.context('Solarize_Light2'):
    for k,v in metrics.items():
        x = np.arange(len(v))+1
        plt.plot(x, v, linewidth=2)

    plt.legend(title='metrics', labels=metrics.keys())
    #plt.legend(title='metrics', title_fontsize = 13, labels=metrics.keys())
    tl = metrics["val_loss"]
    if len(tl) > 10:
        ma = np.percentile(tl,90)
        plt.ylim(top=ma)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("epochs")
    plt.savefig(filename)