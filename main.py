import streamlit as st
import joblib
import numpy as np
from utils import process_new

## Load the model

model=joblib.load('knn_model.pkl')


def Restaurant_classification():
    ## Title
    st.title('Restaurant Classification Project')
    st.markdown('---')

    ## input fields
    
    ## input fields
    name=st.text_input("Kindly,Enter your name")
    online_order=st.selectbox('Did The Restaurant Have Online Ordering Service?',options=[0,1])
    book_table=st.selectbox('Did The Restaurant have Table Booking?',options=[0,1])
    votes=st.number_input('How many votes did the Restaurant have?',step=1)
    location=st.text_input('Where is the Restaurant Located?')
    rest_type=st.text_input('What is the Restaurant Type?')
    cuisines=st.text_input('What is The Restaurant Cuisine?')
    cost=st.number_input('What is the approx cost for two people in the Restaurant',step=1)
    type=st.text_input('What is the Restaurant listed in type?')
    city=st.text_input('What City is the Restaurant in?')

   
    st.markdown('---')

    if st.button('Classify whether the restaurant is good or not.'):
        new_data=np.array([name,online_order,book_table,votes,location,rest_type,cuisines,cost,type,city])

        X_processed=process_new(x_new=new_data)

    ## Predict
        y_pred=model.predict(X_processed)
        if y_pred == 1:
           y_pred ='An Excellent Restaurant'
        else:
            y_pred='Not a well rated Restaurant'
    ## Display
        st.success(f'Your Restaurant is rated as: {y_pred} ')
    
    return None



if __name__=='__main__':
    Restaurant_classification()