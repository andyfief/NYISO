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

        #barplot of the average in each unique value of column
        case "categoryAverage":
            category_avgs = df.groupby(str(column))[str(Y)].mean()
            barPlot(category_avgs.index, category_avgs.values, str(column), str(Y))

        #Number of 1s, 0s in a boolean column
        case "oneHot":
            df['Yes'] = df[str(column)] == 1
            df['No'] = df[str(column)] == 0
            print(f'Number of Yes: {df['Yes'].sum()}')
            print(f'Number of No: {df['No'].sum()}')

        #Check the averages of 0 and 1 in a boolean column
        case "oneHotAvg":
            yes_mask = df[str(column)] == 1
            no_mask = df[str(column)] == 0

            print(f"Yes Mean: {df[yes_mask][str(Y)].mean()}")
            print(f"No Mean: {df[no_mask][str(Y)].mean()}")

        # plot X compared to Y
        case "scatterPlot":
            scatterPlot(df[str(column)], Y, str(column), Y.name)

def main():
    df = pd.read_csv('./data/final_processed_data.csv')
    df_sample = df.sample(frac=0.05, random_state=42)
    
    print((df['christmasDay'] == 1).sum())
    checkRelevance(df_sample, 'Hour', 'Load', 'oneHotAvg')
    checkRelevance(df_sample, 'Day', 'Load', 'categoryAverage')
    
if __name__ == "__main__":
    main()
