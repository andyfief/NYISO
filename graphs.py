import matplotlib.pyplot as plt
import pandas as pd

def averageColumn(column):
    return column / len(column)

def scatterPlot(xVar, yVar, xLabel, yLabel):
    plt.scatter(xVar, yVar, color='blue', alpha=0.7)  
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.show()

def barPlot(xCategories, yVar, xLabel, yLabel):
    plt.bar(xCategories, yVar, color='skyblue', alpha=0.7)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.show()

def checkRelevance(df, column, Y, type):
    match type:
        case "categoryAverage":
            df['averaged'] = averageColumn(df[str(column)])
            category_avgs = df.groupby(str(column))[Y.name].mean()
            barPlot(category_avgs.index, category_avgs.values, str(column), Y.name)

        case "oneHot":
            df['Yes'] = df[str(column)] == 1
            df['No'] = df[str(column)] == 0
            barPlot(['Yes', 'No'], [df['Yes'].sum(), df['No'].sum()], str(column), "Count")

        case "scatterPlot":
            scatterPlot(df[str(column)], Y, str(column), Y.name)

def main():
    df = pd.read_csv('./data/final_processed_data.csv')
    df_sample = df.sample(frac=0.05, random_state=42)

    checkRelevance(df_sample, 'Hour', df_sample['Load'], 'categoryAverage')
    checkRelevance(df_sample, 'christmasDay', df_sample['Load'], 'oneHot')
    checkRelevance(df_sample, 'Hour', df_sample['Load'], 'scatterPlot')

if __name__ == "__main__":
    main()
