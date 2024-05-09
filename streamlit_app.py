import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def load_data():
    df = pd.read_csv("/Users/yifan_/Desktop/Classes/ML/final/mushrooms 2.csv")
    selected_features = [
    'class', 'cap-shape', 'cap-surface', 'bruises', 'gill-attachment', 
    'gill-spacing', 'stalk-shape', 
    'ring-number', 'ring-type', 'habitat'
]
    return df[selected_features]

def main():
    if 'page' not in st.session_state:
        st.session_state.page = 'home'

    if st.session_state.page == 'home':
        st.title("Safe to eat or deadly poison?")
        st.write("Imagine you are on a trip to Yunnan. You accidentally see a mushroom and you want to eat but you don't know if it is poisonous or not.")
        if st.button("Is your mushroom edible or poisonous? 🍄"):
            st.session_state.page = 'classify'
    
    if st.session_state.page == 'classify':
        st.sidebar.title("Mushroom Classification")
        st.sidebar.markdown("Describe your mushroom")

        df = load_data()
        label_encoder = LabelEncoder()
        scaler= StandardScaler()
        encoded_df = df.apply(lambda x: label_encoder.fit_transform(x) if x.dtype == 'O' else x)

        X = encoded_df.drop(['class'], axis=1)
        Y = encoded_df[['class']]
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        y_array_train = y_train['class']
        scaler = StandardScaler().fit(X_train)
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns)
        rf_classifier = RandomForestClassifier(max_depth=5, n_estimators=50)
        rf_classifier.fit(X_train, y_array_train)

        cap_shape = st.sidebar.selectbox("Cap Shape", ('convex', 'bell', 'sunken', 'flat', 'knobbed', 'conical'))
        cap_surface = st.sidebar.selectbox("Cap Surface", ('smooth', 'scaly', 'fibrous', 'grooves'))
        # cap_color = st.sidebar.selectbox("Cap Color", ('brown', 'yellow', 'white', 'gray', 'red', 'pink', 'buff', 'purple', 'cinnamon', 'green'))
        bruises = st.sidebar.selectbox("Bruises", ('Yes', 'No'))
        gill_attachment = st.sidebar.selectbox("Gill Attachment", ('free', 'attached'))
        gill_spacing = st.sidebar.selectbox("Gill Spacing", ('closed', 'crowded'))
        # gill_color = st.sidebar.selectbox("Gill Color", ('black', 'brown', 'gray', 'pink', 'white', 'chocolate', 'purple', 'red', 'buff', 'red', 'yellow', 'orange'))
        stalk_shape = st.sidebar.selectbox("Stalk Shape", ('enlarging', 'tapering'))
        # stalk_color_above_ring = st.sidebar.selectbox("Stalk Color Above Ring", ('white', 'gray', 'pink', 'brown', 'buff', 'red', 'orange', 'cinnamon', 'yellow'))
        # stalk_color_below_ring = st.sidebar.selectbox("Stalk Color Below Ring", ('white', 'gray', 'pink', 'brown', 'buff', 'red', 'orange', 'cinnamon', 'yellow'))
        ring_number = st.sidebar.selectbox("Ring Number", ('one', 'two', 'none'))
        ring_type = st.sidebar.selectbox("Ring Type", ('pendant', 'evanescent', 'large', 'flaring', 'none'))
        # veil_color = st.sidebar.selectbox("Veil Color", ('white', 'brown', 'orange', 'yellow'))
        habitat = st.sidebar.selectbox("Habitat", ('urban', 'grass', 'meadows', 'woods', 'paths', 'waste', 'leaves'))
        threshold = (st.sidebar.number_input("Change Threshold", 50, 100, step=5, value=50, key='threshold'))/100

        cap_shape_mapping = {'bell': 'b', 'conical': 'c', 'convex': 'x', 'flat': 'f', 'knobbed': 'k', 'sunken': 's'}
        cap_surface_mapping = {'fibrous': 'f', 'grooves': 'g', 'scaly': 'y', 'smooth': 's'}
        bruises_mapping = {'Yes': 't', 'No': 'f'}
        gill_attachment_mapping = {'attached': 'a', 'free': 'f'}
        gill_spacing_mapping = {'closed': 'c', 'crowded': 'w'}
        stalk_shape_mapping = {'enlarging': 'e', 'tapering': 't'}
        ring_number_mapping = {'one': 'o', 'two': 't', 'none': 'n'}
        ring_type_mapping = {'pendant': 'p', 'evanescent': 'e', 'large': 'l', 'flaring': 'f', 'none': 'n'}
        habitat_mapping = {'urban': 'u', 'grass': 'g', 'meadows': 'm', 'woods': 'd', 'paths': 'p', 'waste': 'w', 'leaves': 'l'}

        if st.sidebar.button("Predict"):
            new_data = {
             'cap-shape': [cap_shape_mapping[cap_shape]],
             'cap-surface': [cap_surface_mapping[cap_surface]],
             'bruises': [bruises_mapping[bruises]],
               'gill-attachment': [gill_attachment_mapping[gill_attachment]],
               'gill-spacing': [gill_spacing_mapping[gill_spacing]],
               'stalk-shape': [stalk_shape_mapping[stalk_shape]],
              'ring-number': [ring_number_mapping[ring_number]],
              'ring-type': [ring_type_mapping[ring_type]],
              'habitat': [habitat_mapping[habitat]]}


            new_df = pd.DataFrame(new_data)
            # df = df.append(new_df, ignore_index=True)
            df = pd.concat([df, new_df], ignore_index=True)
            encoded_new_df=df.apply(lambda x: label_encoder.fit_transform(x) if x.dtype == 'O' else x)
            encoded_new_df = encoded_new_df.drop(['class'], axis=1)
            scaled_new_df = pd.DataFrame(scaler.fit_transform(encoded_new_df), columns=encoded_new_df.columns)
            scaled_new_df=scaled_new_df.iloc[-1:]
            # prediction = rf_classifier.predict(scaled_new_df)
            proba_predictions = rf_classifier.predict_proba(scaled_new_df)
            prediction = (proba_predictions[:, 0] > threshold).astype(int)
            if prediction == 1:
                st.write("Predicted Class: Edible")
                st.write("Probability: {:.2f}%".format(proba_predictions[0][0] * 100))
                image_url = "https://static.vecteezy.com/system/resources/previews/013/789/428/original/mushroom-cartoon-drawing-on-white-two-brown-cute-mushrooms-icon-for-design-vector.jpg"
                st.image(image_url, use_column_width=True)
            elif prediction == 0:
                st.write("Predicted Class: Poisonous")
                st.write("Probability: {:.2f}%".format(proba_predictions[0][1] * 100))
                image_url = "https://t3.ftcdn.net/jpg/02/04/49/00/360_F_204490094_bSZ9cV18sNih6BLMgHOcwswRBjOVqapI.jpg"
                st.image(image_url, use_column_width=True)

if __name__ == '__main__':
    main()