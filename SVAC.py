import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
import xgboost as xgb
from xgboost import XGBClassifier
from joblib import load
import scikitplot as skplt
import plotly.graph_objects as go


df = pd.read_excel('SVAC_3.2_complete.xlsx')

st.title("Sexual Violence in Armed Conflicts (SVAC)")
st.sidebar.title("Table of contents")
pages=["Home", "DataVizualization", "Modelling", "Analysis", "Perspectives", "About"]
page=st.sidebar.radio("Go to", pages)


# exploration page
if page == pages[0] : 
  st.write("### Context")
  st.write("### Problematic")
  st.write("### Aim")
  st.write("### Presentation of data")
  st.markdown("<h3 style='color:blue; font-weight:bold;'>SVAC Dataset</h3>", unsafe_allow_html=True)
  st.dataframe(df.head(10))
  st.write(df.shape)
  st.dataframe(df.describe())
  if st.checkbox("Show NA") :
    st.dataframe(df.isna().sum())


# data visualization page
if page == pages[1] : 
    st.write("### DataVizualization")

    relevant_columns = ['actor', 'state_prev', 'ai_prev', 'hrw_prev', 'child_prev'] # Focus on the relevant columns for analysis: actor, state_prev, ai_prev, hrw_prev, child_prev
    df_relevant = df[relevant_columns]
  
    df_relevant = df_relevant.replace(-99, np.nan) # Replace -99 with NaN to ensure missing data is handled appropriately
  
    df['sexual_violence'] = df_relevant.apply(lambda x: 1 if any((x == 1) | (x == 2) | (x == 3)) else 0, axis=1) # Create a binary column 'sexual_violence' where 1 indicates any form of sexual violence (1, 2, or 3), and 0 indicates no violence (0)


    # Create a binary variable 'conflict' based on whether there was a conflict in that year
    # We assume that a conflict is indicated if the 'conflictyear' column is 1
    df['conflict'] = df['conflictyear'].apply(lambda x: 1 if x == 1 else 0)

    # Distribution of target
    sexual_violence_counts = df['sexual_violence'].value_counts()
    fig = plt.figure(figsize=(3,3))
    plt.pie(sexual_violence_counts, labels=sexual_violence_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Distribution of Sexual Violence Observations')
    st.pyplot(fig)


    df_filtered = df[(df_relevant != -99).all(axis=1)]
    # Count the total number of conflicts per year
    total_conflicts = df_filtered.groupby('year').size()

    # Count the number of conflicts with any sexual violence (where sexual_violence is 1, 2, or 3)
    conflicts_with_sv = df_filtered[df_filtered['sexual_violence'].isin([1, 2, 3])].groupby('year').size()

    # Combine these into a DataFrame
    data = pd.DataFrame({
    'Total Conflicts': total_conflicts,
    'Conflicts with Sexual Violence': conflicts_with_sv}).fillna(0)  # Fill NaN values with 0

    # Step 2: Create Stacked Bar Chart
    fig_2 = plt.figure(figsize=(12, 8))

    # Plot the total number of conflicts in yellow
    plt.bar(data.index, data['Total Conflicts'], color='yellow', label='Total Conflicts')

    # Plot the number of conflicts with sexual violence in red, stacked on top of the yellow bars
    plt.bar(data.index, data['Conflicts with Sexual Violence'], color='red', label='Conflicts with Sexual Violence')

    # Add labels and title
    plt.title('Development of Conflicts and Sexual Violence in Conflicts by Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Conflicts')
    plt.legend(title='Conflict Data')
    plt.grid(True)
    st.pyplot(fig_2)


 

    # Find the top 10 actors with the most and least documented cases of sexualized violence
    top_10_most = df_filtered.groupby('actor')['sexual_violence'].sum().nlargest(10).index
    top_10_least = df_filtered.groupby('actor')['sexual_violence'].sum().nsmallest(10).index

    # Filter the data for these actors
    top_actors_most = df_filtered[df_filtered['actor'].isin(top_10_most)]
    top_actors_least = df_filtered[df_filtered['actor'].isin(top_10_least)]

    # Prepare data for plotting: Yearly summary for each actor
    yearly_data_most = top_actors_most.groupby(['actor', 'year']).agg({'conflictyear': 'sum','sexual_violence': 'sum'}).reset_index()

    yearly_data_least = top_actors_least.groupby(['actor', 'year']).agg({'conflictyear': 'sum','sexual_violence': 'sum'}).reset_index()

    # Create traces for each actor (one trace for conflicts and one for sexual violence)
    traces = []

    # Adding traces for the top 10 actors with the most cases
    for actor in top_10_most:
        actor_data = yearly_data_most[yearly_data_most['actor'] == actor]
        traces.append(go.Scatter(x=actor_data['year'], y=actor_data['conflictyear'],mode='lines+markers', name=f'Conflict - {actor}', visible=True))
        traces.append(go.Scatter(x=actor_data['year'], y=actor_data['sexual_violence'],mode='lines+markers', name=f'Violence - {actor}', visible=True))

    # Adding traces for the top 10 actors with the least cases
    for actor in top_10_least:
        actor_data = yearly_data_least[yearly_data_least['actor'] == actor]
        traces.append(go.Scatter(x=actor_data['year'], y=actor_data['conflictyear'],mode='lines+markers', name=f'Conflict - {actor}', visible=True))
        traces.append(go.Scatter(x=actor_data['year'], y=actor_data['sexual_violence'], mode='lines+markers', name=f'Violence - {actor}', visible=True))

    # Create the figure
    fig_3 = go.Figure(data=traces)

    # Update layout to add dropdowns for each actor
    buttons = []
    for i, actor in enumerate(top_10_most.tolist() + top_10_least.tolist()):
        button = dict(label=actor,method="update",args=[{"visible": [j == i * 2 or j == i * 2 + 1 for j in range(len(traces))]},
                                                        {"title": f"Conflict and Sexual Violence for {actor}"}])
        buttons.append(button)

    fig_3.update_layout(updatemenus=[dict(active=0,buttons=buttons,direction="down",showactive=True,),],
                      width=1600,height=1100,
                      title="Top 10 Actors: Conflict Duration and Sexualized Violence",
                      xaxis_title="Year",
                      yaxis_title="Count")
    
    st.plotly_chart(fig_3)



# modelling page
if page == pages[2] : 
    st.write("### Modelling")
    st.markdown("Sexual Violence :red[Prediction]")
  

  # importing trained models
    rf = load('random_forest_model.joblib')
    xgb_model = load('xgboost_model.joblib')
  
  # Feature-Selection
    selected_features = ['conflictyear', 'conflict', 'incomp', 'region', 'actor', 'gwnoloc', 'actor_type', 'interm', 'postc']
  
  
  # Prep Data
    X = df[selected_features]
    y = df['sexual_violence']
  
  
  # One-Hot-Encoding for categorial variables
    categorical_columns = ['conflict', 'region', 'actor', 'gwnoloc', 'actor_type', 'interm', 'postc']
    for col in categorical_columns:
            X.loc[:, col] = X[col].astype(str)
    
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(drop='first'), categorical_columns)], remainder='passthrough')

    X_encoded = preprocessor.fit_transform(X)

    # Train and Testset
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42, stratify=y)

    # Over-Sampling
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

    choice = ['Random Forest', 'XGBoost']
    option = st.selectbox('Choice of the model', choice)
    st.write('The chosen model is :', option)

    def prediction(classifier):
        if classifier == 'Random Forest':
            clf = rf
        elif classifier == 'XGBoost':
            clf = xgb_model
        clf.fit(X_resampled, y_resampled)
        return clf
    

    clf = prediction(option)

    tab1 = st.tabs(["Global Performance"])[0]

    with tab1:
        st.header("Confusion Matrix | Feature Importances")
        col1, col2 = st.columns(2)
        with col1:
            conf_mat_fig = plt.figure(figsize=(6,6))
            ax1 = conf_mat_fig.add_subplot(111)
            skplt.metrics.plot_confusion_matrix(y_test, clf.predict(X_test), ax=ax1, normalize=True)
            st.pyplot(conf_mat_fig, use_container_width=True)

        with col2:
            feat_imp_fig = plt.figure(figsize=(6,6))
            ax1 = feat_imp_fig.add_subplot(111)
            skplt.estimators.plot_feature_importances(clf, feature_names=np.array(preprocessor.get_feature_names_out()), ax=ax1, x_tick_rotation=90)
            st.pyplot(feat_imp_fig, use_container_width=True)

        st.divider()
        st.header("Classification Report")
        st.code(classification_report(y_test, clf.predict(X_test)))