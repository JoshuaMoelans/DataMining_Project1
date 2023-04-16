import seaborn as sns
from matplotlib import pyplot as plt


def visualize_columns(data):
    """
    Visualize the columns of the data; looks a bit messy using matplotlib, but it's a start
    :param data:
    :return:
    """
    # visualize each column using a histogram
    feature_cols = ['age', 'workclass', 'education', 'education-num', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
    for col in feature_cols:
        data[col].hist(rwidth=0.7)
        plt.title(col)
        plt.xticks(rotation=90)  # rotate x-axis labels
        plt.grid(visible=None)  # remove grid
        # set bottom margin based on column name length
        if col in ['marital-status', 'native-country', 'education', 'occupation', 'relationship', 'workclass',
                   'capital-gain', 'capital-loss']:
            plt.subplots_adjust(bottom=0.4)
        else:
            plt.subplots_adjust(bottom=0.1)
        plt.savefig("./results/columns/" + col + ".png")
        plt.clf()


def visualize_columns_2(data):
    """
    Second attempt at visualizing the columns of the data; looks better using seaborn
    :param data: data to visualize; expected to have all feature_cols as columns
    :return:
    """
    feature_cols = ['age', 'workclass', 'education', 'education-num', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
    for col in feature_cols:
        if col in ['marital-status', 'native-country', 'education', 'occupation', 'relationship', 'workclass'
            , 'race']:
            plt.subplots_adjust(bottom=0.5)
            plt.xticks(rotation=90)
            sns.countplot(x=col, data=data)  # actually plot the data
        if col in ['age', 'hours-per-week', 'capital-gain', 'capital-loss']:
            # toDo find ways to properly show these features
            sns.histplot(data=data, x=col)
            plt.subplots_adjust(bottom=0.1)
        plt.savefig("./results/columns/" + col + ".png")
        plt.clf()
        plt.cla()

    # visualize pairplot
    # sns.pairplot(data, hue='class', corner=True)
    # plt.savefig("./results/columns/pairplot.png")
    # plt.clf()
