import streamlit as st
import cv2
from PIL import Image

from utils import preprocess_img, detect_contours, evaluate

st.set_page_config(layout='wide')

st.title("Handwritten Equation Solver")

col1, col2 = st.columns(2)

with col1:
    st.header("Training")
    st.info("The dataset consisted of 16 classes. Each image was 28x28")
    st.text("Class Labels: '%', '*','+','-','0','1','2','3','4','5','6','7','8','9','[',']'")
    st.image("/3_1_0.png", width=200)

    st.info("The model was trained on a 80/20 split and achieved 98% validation accuracy in 10 epochs")

    st.header("Recognition and Prediction")
    st.text("This is the image we will use for explanation")
    image = cv2.imread('/img_3.png')
    st.image(image, width=800)

    binary_image = preprocess_img(image)
    st.info("We then convert the image to grayscale, smooth it and binarize it")
    st.image(binary_image, width=800)


    st.info("We the detect contours to extract operators and operands")
    chars, result, img_marked = detect_contours(image, binary_image)
    st.image(img_marked, width=800)

    # Calculate the number of rows needed for the 2-column grid
    num_rows = len(chars) // 2 + len(chars) % 2

    # Display elements in a 2-column grid
    for i in range(num_rows):
        col11, col12 = st.columns(2)

        # Display the element in the first column
        with col11:
            st.image(chars[i * 2], width=100)

        # Display the element in the second column if it exists
        if i * 2 + 1 < len(chars):
            with col12:
                st.image(chars[i * 2 + 1], width=100)

    st.info("The results are then fed into the model which then classifies each operator or operand")
    st.text(f"Results: {result}")
    st.text("We can see that our model misclassifies '1' as '-' ")

    st.info("The results are then concatenated into a string and solved using 'Numexpr'")
    st.text(evaluate(result))




with col2:
    st.header("Try with you expression")

    # File uploader allows user to add their own image
    uploaded_file = st.file_uploader("Upload an image of the arithmetic equation", type=["png", "jpg", "jpeg"])

    # When a file is uploaded
    if uploaded_file is not None:
        # Convert the file to an image
        img = Image.open(uploaded_file)
        img = img.save("img_usr.png")

        # OpenCv Read
        img = cv2.imread("img_usr.png")

        # Display the uploaded image
        st.image(img, caption='Uploaded Equation', width=800)

        # Preprocess and predict (put your own preprocess and predict functions here)
        # prediction = predict(image)

        # Display the prediction
        # st.write(f'The solved equation is: {prediction}')

        # Add a button to solve the equation
        if st.button('Solve Equation'):
            binary_image_user = preprocess_img(img)
            st.info("We then convert the image to grayscale, smooth it and binarize it")
            st.image(binary_image_user, width=800)

            st.info("We the detect contours to extract operators and operands")
            chars_user, result_user, img_marked_user = detect_contours(img, binary_image_user)
            st.image(img_marked_user, width=800)

            # Calculate the number of rows needed for the 2-column grid
            num_rows_user = len(chars_user) // 2 + len(chars_user) % 2

            # Display elements in a 2-column grid
            for i in range(num_rows_user):
                col21, col22 = st.columns(2)

                # Display the element in the first column
                with col21:
                    st.image(chars_user[i * 2], width=100)

                # Display the element in the second column if it exists
                if i * 2 + 1 < len(chars_user):
                    with col22:
                        st.image(chars_user[i * 2 + 1], width=100)

            st.info("The results are then fed into the model which then classifies each operator or operand")
            st.text(f"Results: {result_user}")

            st.info("The results are then concatenated into a string and solved using 'Numexpr'")
            st.text(evaluate(result_user))

    # Process the image and solve the equation
    # solved_eq = solve_equation(image)

    # Display the solution
    # st.write(f'The solution is: {solved_eq}')
