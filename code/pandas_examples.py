import pandas as pd
import sys
import pandas as pd
import numpy as np


def import_data(fname):
    df=pd.read_csv(fname)
    #print(df.head(3))
    #survived=df[:100].Survived.values
    #survived=df[:100]["Survived"]
    #survived=df[:100][["Survived","Name","Age"]]
    #age= df[df["Age"].isin(range(25,35))]
    #age= df[df["Age"].isin(range(25,35))]["Name"]
    #print(age)
    return df


def check_null_values(df,cname):
    print(df[cname].notna())
    print(df[cname].isna())

def select_non_null_values(df,cname):
    print(df[df[cname].notna()][cname])

def create_dataframe(value_matrix, column_labels):
    df = pd.DataFrame(value_matrix, columns=column_labels)
    return df

def main():
    df=import_data("titanic.csv")
    #check_null_values(df,"Age")
    #select_non_null_values(df,"Age")
    df=create_dataframe([[np.nan, "a", np.nan, 0],
                          [3, "b", np.nan, 1],
                          [np.nan, "c", np.nan, 5],
                          [np.nan, np.nan, np.nan, 4]],
                          list('ABCD'))
    print(df)
    df_copy=df.copy()
    #df_copy=df_copy.fillna(0) # fill all NaN with 0
    #print("Zero -filled:",df_copy)
    df_copy=df_copy["B"].fillna("UNKNOWN")
    print(df_copy)


if __name__=="__main__":
    main()

