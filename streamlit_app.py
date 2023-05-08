import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set a title for your app
st.title("Sample Streamlit App")

# Add a button
button_clicked = st.button("Click me!")

# Check if the button is clicked
if button_clicked:
    # Generate some random data
    data = np.random.randn(100, 2)

    # Create a DataFrame
    df = pd.DataFrame(data, columns=["X", "Y"])

    # Display the DataFrame
    st.write(df)

    # Create a scatter plot using matplotlib
    plt.scatter(df["X"], df["Y"])
    st.pyplot(plt)