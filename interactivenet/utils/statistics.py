import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def ResultPlot(data, scorename="Dice", types=False, unseen=False):
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    fig, ax = plt.subplots()

    data = pd.DataFrame.from_dict(data, orient="index")
    data.columns = [scorename]
    data["Names"] = data.index
    data = data.reset_index()

    if types:
        data['Types'] = data['Names'].map(types)
        if unseen:
            data['Seen in training'] = data['Types'].map(unseen)
            data.loc[data['Seen in training'] != False, 'Seen in training'] = True
            sns.boxplot(x="Types", y=scorename, hue="Seen in training", dodge=False, data=data, ax=ax)
        else:
            sns.boxplot(x="Types", y=scorename, data=data, ax=ax)
    else:
        sns.boxplot(y=scorename, data=data, ax=ax)

    plt.xticks(rotation = 45, ha="right", rotation_mode="anchor")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    return fig
