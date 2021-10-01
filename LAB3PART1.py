from matplotlib import pyplot as plt
from PIL import Image
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import time
import seaborn as sns

from PIL import Image


def myDecorator(function):
    def modified_function(df):
        time_ = time.time()
        res = function(df)
        time_ = time.time()-time_
        with open(f"{function.__name__}_exec_time.txt","w") as f:
            f.write(f"{time_}")
        return res
    return modified_function


@st.cache
def load_data(path):
    df = pd.read_csv(path)[:10000]
    return df


@myDecorator
@st.cache
def df1_data_transformation(df_):
    df = df_.copy()
    df["Date/Time"] = df["Date/Time"].map(pd.to_datetime)

    def get_dom(dt):
        return dt.day
    def get_weekday(dt):
        return dt.weekday()
    def get_hours(dt):
        return dt.hour

    df["weekday"] = df["Date/Time"].map(get_weekday)
    df["dom"] = df["Date/Time"].map(get_dom)
    df["hours"] = df["Date/Time"].map(get_hours)

    return df


@myDecorator
@st.cache
def df2_data_transformation(df_):
    df = df_.copy()
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])

    def get_hours(dt):
        return dt.hour

    df["hours_pickup"] = df["tpep_pickup_datetime"].map(get_hours)
    df["hours_dropoff"] = df["tpep_dropoff_datetime"].map(get_hours)

    return df

@st.cache(allow_output_mutation=True)
def frequency_by_dom(df):
    fig, ax = plt.subplots(figsize=(10,6))
    ax.set_title("Frequency by DoM - Uber - April 2014")
    ax.set_xlabel("Date of the month")
    ax.set_ylabel("Frequency")
    ax = plt.hist(x=df.dom, bins=30, rwidth=0.8, range=(0.5,30.5))
    return fig

@st.cache
def map_data(df):
    df_ = df[["Lat","Lon"]]
    df_.columns=["lat","lon"]
    return df_

@st.cache(allow_output_mutation=True)
def data_by(by,df):
    def count_rows(rows):
        return len(rows)
    
    if by == "dom":
        fig, ax = plt.subplots(1,2, figsize=(10,6))
        ax[0].set_ylim(40.72,40.75)
        ax[0].bar(x=sorted(set(df["dom"])),height=df[["dom","Lat"]].groupby("dom").mean().values.flatten())
        ax[0].set_title("Latitude moyenne par jour du mois")

        ax[1].set_ylim(-73.96,-73.98)
        ax[1].bar(x=sorted(set(df["dom"])),height=df[["dom","Lon"]].groupby("dom").mean().values.flatten(), color="orange")
        ax[1].set_title("Longitude moyenne par jour du mois")
        return fig
    
    elif by == "hours":
        fig, ax= plt.subplots(figsize=(10,6))
        ax = plt.hist(x=df.hours, bins=24, range=(0.5,24))
        return fig
    
    elif by == "dow":
        fig, ax= plt.subplots(figsize=(10,6))
        ax = plt.hist(x=df.weekday, bins=7, range=(-5,6.5))
        return fig
    
    elif by == "dow_xticks":
        fig, ax= plt.subplots(figsize=(10,6))
        ax.set_xticklabels('Mon Tue Wed Thu Fri Sat Sun'.split())
        ax.set_xticks(np.arange(7))
        ax = plt.hist(x=df.weekday, bins=7, range=(0,6))
        return fig
    
    else:
        pass

@st.cache
def group_by_wd(df):    
    def count_rows(rows):
        return len(rows)
    grp_df = df.groupby(["weekday","hours"]).apply(count_rows).unstack()
    return grp_df

@st.cache(allow_output_mutation=True)
def grp_heatmap(df):
    fig, ax= plt.subplots(figsize=(10,6))
    ax = sns.heatmap(grp_df)
    return fig

@st.cache(allow_output_mutation=True)
def lat_lon_hist(df,fusion=False):
    lat_range = (40.5,41)
    lon_range = (-74.2,-73.6)

    if fusion:
        fig, ax = plt.subplots()
        ax1 = ax.twiny()
        ax.hist(df.Lon, range=lon_range, color="yellow")
        ax.set_xlabel("Latitude")
        ax.set_ylabel("Frequency")

        ax1.hist(df.Lat, range=lat_range)
        ax1.set_xlabel("Longitude")
        ax1.set_ylabel("Frequency")
        return fig
    
    else:
        fig, ax = plt.subplots(1,2, figsize=(10,5))


        ax[0].hist(df.Lat, range=lat_range, color="red")
        ax[0].set_xlabel("Latitude")
        ax[0].set_ylabel("Frequence")

        ax[1].hist(df.Lon, range=lon_range, color="green")
        ax[1].set_xlabel("Longitude")
        ax[1].set_ylabel("Frequence")
        return fig

@st.cache(allow_output_mutation=True)
def display_points(data, color=None):
    fig, ax= plt.subplots(figsize=(10,6))
    ax = sns.scatterplot(data=data) if color == None else sns.scatterplot(data=data, color=color)
    return fig

@st.cache(allow_output_mutation=True)
def passengers_graphs_per_hour(df):
    fig, ax = plt.subplots(2,2, figsize=(10,6))

    for ax_ in ax:
        for ax__ in ax_:
            ax__.set_xticks(np.arange(24))

    ax[0,0].bar(x=sorted(set(df["hours_pickup"])), height=df[["hours_pickup","passenger_count"]].groupby("hours_pickup").sum().values.flatten(), color="red")
    ax[0,0].set_title("Total Number of passengers per pickup hour")

    ax[0,1].bar(x=sorted(set(df["hours_pickup"])), height=df[["hours_pickup","passenger_count"]].groupby("hours_pickup").mean().values.flatten(), color="yellow")
    ax[0,1].set_title("Average Number of passengers per pickup hour")

    ax[1,0].bar(x=sorted(set(df["hours_pickup"])), height=df["hours_pickup"].value_counts().sort_index().values.flatten(), color="green")
    ax[1,0].set_title("Total number of passages per pickup hour")
    return fig

@st.cache(allow_output_mutation=True)
def passengers_graphs_per_dropoff_hour(df):
    fig, ax = plt.subplots(2,2, figsize=(12,6))

    for ax_ in ax:
        for ax__ in ax_:
            ax__.set_xticks(np.arange(24))

    ax[0,0].bar(x=sorted(set(df["hours_dropoff"])), height=df[["hours_dropoff","passenger_count"]].groupby("hours_dropoff").sum().values.flatten())
    ax[0,0].set_title("Total Number of passengers per dropoff hour")

    ax[0,1].bar(x=sorted(set(df["hours_dropoff"])), height=df[["hours_dropoff","passenger_count"]].groupby("hours_dropoff").mean().values.flatten(), color="black")
    ax[0,1].set_title("Average Number of passengers per dropoff hour")

    ax[1,0].bar(x=sorted(set(df["hours_dropoff"])), height=df["hours_dropoff"].value_counts().sort_index().values.flatten(), color="orange")
    ax[1,0].set_title("Total number of passages per dropoff hour")
    return fig

@st.cache(allow_output_mutation=True)
def amount_graphs_per_hour(df):
    fig, ax = plt.subplots(1,2, figsize=(12,6))

    for ax_ in ax:
        ax_.set_xticks(np.arange(24))

    ax[0].bar(x=sorted(set(df["hours_pickup"])), height=df[["hours_pickup","trip_distance"]].groupby("hours_pickup").sum().values.flatten(), color="grey")
    ax[0].set_title("Total trip distance per pickup hour")

    ax[1].bar(x=sorted(set(df["hours_pickup"])), height=df[["hours_pickup","trip_distance"]].groupby("hours_pickup").mean().values.flatten())
    ax[1].set_title("Average trip distance per pickup hour")
    return fig

@st.cache(allow_output_mutation=True)
def distance_graphs_per_hour(df):
    fig, ax = plt.subplots(1,2, figsize=(10,6))

    for ax_ in ax:
        ax_.set_xticks(np.arange(24))

    ax[0].bar(x=sorted(set(df["hours_pickup"])), height=df[["hours_pickup","total_amount"]].groupby("hours_pickup").sum().values.flatten(), color="lime")
    ax[0].set_title("Total amount per hour")

    ax[1].bar(x=sorted(set(df["hours_pickup"])), height=df[["hours_pickup","total_amount"]].groupby("hours_pickup").mean().values.flatten(), color="pink")
    ax[1].set_title("Average amount per hour")
    return fig

@st.cache(allow_output_mutation=True)
def corr_heatmap(df):
    fig, ax = plt.subplots(figsize=(10,6))
    ax = sns.heatmap(df.corr())
    return fig
    
image = Image.open('uber-logo.jpg')
st.image(image, caption=' ', width = 200) 
st.title(" Dashboard For Uber ")
st.markdown(" This dashboard will visualize some data related to Uber ")
st.markdown(" Uber longitute and latitude data")
st.sidebar.title("Visualization Selector")
st.sidebar.markdown("Select the Charts/Plots accordingly:")


st.header('Load data')
df = pd.read_csv("uber-raw-data-apr14.csv")
if st.checkbox('Show dataframe'):
    st.write(df)
st.header('Exploring the Dataset')
st.write('head of data')
if st.checkbox('Show 10 first rows'):
    st.write(df.head(10))
st.write('tail of data')
if st.checkbox('Show 10 last rows'):
    st.write(df.tail(10))
df["Date/Time"] = pd.to_datetime(df["Date/Time"])
def get_dom(dt):
    return dt.day
df['lat'] = df['Lat']
df['lon'] = df["Lon"]

df['dom'] = df['Date/Time'].map(get_dom)


def get_weekday(dt):
    return dt.weekday()

df['weekday'] = df['Date/Time'].map(get_weekday)

st.header('visual representation')
st.subheader('Frequency by day of the month')
plt.hist(df['dom'], bins=30, rwidth=0.8, range=(0.5, 30.5))
plt.title("Frequency by DoM - Uber - April 2014")
plt.xlabel("Date of the month")
plt.ylabel(" Frequency")
st.pyplot()

if st.sidebar.checkbox('show the map of trips'):
    st.write('representation of the trips localisation in ny')
    st.map(df)
if st.sidebar.checkbox('show the heatmap'):
    st.write('heatmap of the trips in the weekdays by hours')
    df['hour'] = df['Date/Time'].dt.hour
    grpby2 = df.groupby(['weekday', 'hour']).apply(count_rows).unstack()
    sns.heatmap(grpby2)
    st.pyplot()



def count_rows(rows):
    return len(rows)
st.subheader('Frequency by weekday')
labels = "Mon Tue Wed Thu Fri Sat Sun".split()
df['groupe'] = df.groupby(df["weekday"]).apply(count_rows)
st.set_option('deprecation.showPyplotGlobalUse', False)
labels = "Mon Tue Wed Thu Fri Sat Sun".split()
grpby2 = df.groupby(df["weekday"]).apply(count_rows)
plt.hist(grpby2)

plt.xticks(grpby2, labels)
st.pyplot()

if st.sidebar.checkbox("show emplacement and Frequency graph"):
    ax1 = sns.histplot(df["Lat"], binrange=(min(df["Lat"]), max(df["Lat"])))
    min(df["Lat"])
    max(df["Lat"])
    ax2 = sns.histplot(df["Lon"], binrange=(min(df["Lon"]), max(df["Lon"])))
    min(df["Lon"])
    max(df["Lon"])
    ax1 = sns.histplot(df["Lat"],binrange=(min(df["Lat"]), max(df["Lat"])))
    fig, ax = plt.subplots(figsize=(20, 20))
    ax2 = ax1.twiny()
    ax2 = sns.histplot(df["Lon"],binrange=(min(df["Lon"]), max(df["Lon"])))

    st.pyplot()
